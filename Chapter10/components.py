"""
This file has vertex ai comoponent definitions,
same components can be executed using the jupyter notebook
from the same folder.
"""

from typing import NamedTuple
import typing
import pandas as pd
from kfp.v2 import dsl
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics,
                        component, 
                        OutputPath, 
                        InputPath)

from kfp.v2 import compiler
from google.cloud import bigquery
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
from google_cloud_pipeline_components import aiplatform as gcc_aip

@component(
    packages_to_install=["pandas", "pyarrow", "scikit-learn==1.0.0"],
    base_image="python:3.9",
    output_component_file="load_data_component.yaml"
)

def get_wine_data(
    url: str,
    dataset_train: Output[Dataset],
    dataset_test: Output[Dataset]
):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split as tts
    
    df_wine = pd.read_csv(url, delimiter=";")
    df_wine['best_quality'] = [1 if x>=7 else 0 for x in df_wine.quality] 
    df_wine['target'] = df_wine.best_quality
    df_wine = df_wine.drop(
        ['quality', 'total sulfur dioxide', 'best_quality'],
         axis=1,
    )
    train, test = tts(df_wine, test_size=0.3)
    train.to_csv(
        dataset_train.path + ".csv",
        index=False, 
        encoding='utf-8-sig',
    )
    test.to_csv(
        dataset_test.path + ".csv",
        index=False,
        encoding='utf-8-sig',
    )

@component(
    packages_to_install = [
        "pandas",
        "scikit-learn"
    ], 
    base_image="python:3.9",
    output_component_file="model_training_component.yml",
)
def train_winequality(
    dataset:  Input[Dataset],
    model: Output[Model], 
):
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import pickle

    data = pd.read_csv(dataset.path+".csv")
    model_rf = RandomForestClassifier(n_estimators=10)
    model_rf.fit(
        data.drop(columns=["target"]),
        data.target,
    )
    model.metadata["framework"] = "RF"
    file_name = model.path + f".pkl"
    with open(file_name, 'wb') as file:  
        pickle.dump(model_rf, file)

@component(
    packages_to_install = [
        "pandas",
        "scikit-learn"
    ], 
    base_image="python:3.9",
    output_component_file="model_evaluation_component.yml",
)
def winequality_evaluation(
    test_set:  Input[Dataset],
    rf_winequality_model: Input[Model],
    thresholds_dict_str: str,
    metrics: Output[ClassificationMetrics],
    kpi: Output[Metrics]
) -> NamedTuple("output", [("deploy", str)]):

    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import logging 
    import pickle
    from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score
    import json
    import typing

    def threshold_check(val1, val2):
        cond = "false"
        if val1 >= val2 :
            cond = "true"
        return cond

    data = pd.read_csv(test_set.path+".csv")
    model = RandomForestClassifier()
    file_name = rf_winequality_model.path + ".pkl"
    with open(file_name, 'rb') as file:  
        model = pickle.load(file)
    
    y_test = data.drop(columns=["target"])
    y_target=data.target
    y_pred = model.predict(y_test)
    
    y_scores =  model.predict_proba(
        data.drop(columns=["target"])
    )[:, 1]
    fpr, tpr, thresholds = roc_curve(
         y_true=data.target.to_numpy(),
        y_score=y_scores, pos_label=True
    )
    metrics.log_roc_curve(
        fpr.tolist(), 
        tpr.tolist(), 
        thresholds.tolist()
    )  
    
    metrics.log_confusion_matrix(
       ["False", "True"],
       confusion_matrix(
           data.target, y_pred
       ).tolist(), 
    )
    
    accuracy = accuracy_score(data.target, y_pred.round())
    thresholds_dict = json.loads(thresholds_dict_str)
    rf_winequality_model.metadata["accuracy"] = float(accuracy)
    kpi.log_metric("accuracy", float(accuracy))
    deploy = threshold_check(float(accuracy), int(thresholds_dict['roc']))
    return (deploy,)

@component(
    packages_to_install=["google-cloud-aiplatform", "scikit-learn",  "kfp"],
    base_image="python:3.9",
    output_component_file="model_winequality_component.yml"
)
def deploy_winequality(
    model: Input[Model],
    project: str,
    region: str,
    serving_container_image_uri : str, 
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model]
):
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=region)

    DISPLAY_NAME  = "winequality"
    MODEL_NAME = "winequality-rf"
    ENDPOINT_NAME = "winequality_endpoint"
    
    def create_endpoint():
        endpoints = aiplatform.Endpoint.list(
        filter='display_name="{}"'.format(ENDPOINT_NAME),
        order_by='create_time desc',
        project=project, 
        location=region,
        )
        if len(endpoints) > 0:
            endpoint = endpoints[0]  # most recently created
        else:
            endpoint = aiplatform.Endpoint.create(
            display_name=ENDPOINT_NAME, project=project, location=region
        )
    endpoint = create_endpoint()   
    
    #Import a model programmatically
    model_upload = aiplatform.Model.upload(
        display_name = DISPLAY_NAME, 
        artifact_uri = model.uri.replace("model", ""),
        serving_container_image_uri =  serving_container_image_uri,
        serving_container_health_route=f"/v1/models/{MODEL_NAME}",
        serving_container_predict_route=f"/v1/models/{MODEL_NAME}:predict",
        serving_container_environment_variables={
        "MODEL_NAME": MODEL_NAME,
    },       
    )
    model_deploy = model_upload.deploy(
        machine_type="n1-standard-4", 
        endpoint=endpoint,
        traffic_split={"0": 100},
        deployed_model_display_name=DISPLAY_NAME,
    )

    # Save data to the output params
    vertex_model.uri = model_deploy.resource_name

@dsl.pipeline(
    pipeline_root=BUCKET_URI,
    name="pipeline-winequality",
)
def pipeline(
    url: str = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
    project: str = PROJECT_ID,
    region: str = REGION, 
    display_name: str = DISPLAY_NAME,
    api_endpoint: str = REGION+"-aiplatform.googleapis.com",
    thresholds_dict_str: str = '{"roc":0.8}',
    serving_container_image_uri: str = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"
    ):
    
    # adding first component
    data_op = get_wine_data(url)
    # second component uses output of first component as input
    train_model_op = train_winequality(data_op.outputs["dataset_train"])
    # add third component (uses outputs of comp1 and comp2 as input)
    model_evaluation_op = winequality_evaluation(
        test_set=data_op.outputs["dataset_test"],
        rf_winequality_model=train_model_op.outputs["model"],
        # We deploy the model only if the model performance is above the threshold
        thresholds_dict_str = thresholds_dict_str, 
    )
    
    # condition to deploy the model
    with dsl.Condition(
        model_evaluation_op.outputs["deploy"]=="true",
        name="deploy-winequality",
    ):      
        deploy_model_op = deploy_winequality(
        model=train_model_op.outputs['model'],
        project=project,
        region=region, 
        serving_container_image_uri = serving_container_image_uri,
        )
