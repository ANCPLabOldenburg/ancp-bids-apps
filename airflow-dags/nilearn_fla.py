from airflow.decorators import dag, task
import pendulum
from airflow.operators.python import get_current_context


def get_var(key):
    ctx = get_current_context()
    val = ctx["dag_run"].conf[key]
    return val


@task()
def load_subjects():
    dataset_path = get_var("dataset_path")
    import ancpbids
    ds = ancpbids.load_dataset(dataset_path)
    subjects = ds.query_entities()["subject"]
    return list(subjects)




@task()
def process_subject(sub_label) -> dict:
    return {}


@dag(start_date=pendulum.now())
def nilearn_fla():
    subjects = load_subjects()
    process_subject.expand(sub_label=subjects)


dag = nilearn_fla()

if __name__ == "__main__":
    dag.test(run_conf={"dataset_path": "c:/work/datasets/fMRI-language-localizer-demo-dataset/"})
