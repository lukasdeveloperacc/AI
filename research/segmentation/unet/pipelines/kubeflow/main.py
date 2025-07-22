from kfp.dsl import container_component, ContainerSpec, pipeline

@container_component
def train():
    return ContainerSpec(
        image="unet:v0.1.0",
        command=["python3", "segmentation/unet/train_with_hydra.py"],
    )

@container_component
def test():
    return ContainerSpec(
        image="unet:v0.1.0",
        command=["python3", "segmentation/unet/test_with_hydra.py"],
    )

@pipeline
def pipeline():
    train_task = train()
    test_task = test()
    
    test_task.after(train_task)
 
if __name__ == "__main__":
    from kfp.compiler import Compiler
    
    Compiler().compile(pipeline, "pipeline.yaml")
