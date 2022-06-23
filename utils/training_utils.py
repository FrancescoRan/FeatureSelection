from re import sub, findall
from os import listdir, mkdir
from os.path import exists, join
from numpy import max
from joblib import dump, load


def _get_model_name(estimator, pipeline: bool = True) -> str:

    if pipeline:
        best_estimator = estimator.best_estimator_
        model_name = str(best_estimator.steps[-1][-1].__str__()).split("(")[0]
        # model_name = sub("[\(\[].*?[\)\]]", "", model_name)

    else:
        model_name = str(estimator.__str__()).replace("\n", "").split("(")[0]
        # model_name = sub("[\(\[].*?[\)\]]", "", model_name)

    return model_name


def _set_name_to_artifact(artifacts_folder: str, model_name: str) -> str:

    if any([model_name in a for a in listdir(artifacts_folder)]):

        max_version = max(
            [
                int(findall("\d+", model)[0])
                for model in listdir(artifacts_folder)
                if model_name in model
            ]
        )
        model_name = model_name + "__v{}.joblib".format(max_version + 1)

    else:
        model_name = model_name + "__v1.joblib"

    return model_name


def dump_artifact(
        pipeline_estimator, artifacts_folder: str, artifact_name: str):

    if not exists(artifacts_folder):
        mkdir(artifacts_folder)

    artifact_name = _set_name_to_artifact(artifacts_folder, artifact_name)
    artifact_path = join(artifacts_folder, artifact_name)
    # with open(model_path, "wb") as m:
    #     pickle.dump(pipeline_estimator, m)
    dump(pipeline_estimator, artifact_path, compress=1)
    return artifact_name

