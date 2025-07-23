from kfp.dsl import container_component, ContainerSpec, pipeline
from config.config import TrainConfig, TestConfig
from typing import Optional

@container_component
def train(
    dataset_config: Optional[str] = None,
    network_config: Optional[str] = None,
    optimizer_config: Optional[str] = None,
    loss_config: Optional[str] = None,
    trainer_config: Optional[str] = None,
):
    command = ["python3", "segmentation/unet/train_with_hydra.py"]
    
    overrides = []
    if dataset_config:
        import json
        config_dict: dict = json.loads(dataset_config)
        overrides.append(f"dataset={config_dict.get('dataset')}")
        config_dict.pop("dataset")
        for k, v in config_dict.items():
            v = str(v).lower() if isinstance(v, bool) else v
            overrides.append(f"dataset.{k}={v}")
    if network_config:
        import json
        config_dict: dict = json.loads(network_config)
        overrides.append(f"network={config_dict.get('network')}")
        config_dict.pop("network")
        for k, v in config_dict.items():
            v = str(v).lower() if isinstance(v, bool) else v
            overrides.append(f"network.{k}={v}")
    if optimizer_config:
        import json
        config_dict: dict = json.loads(optimizer_config)
        overrides.append(f"optimizer={config_dict.get('optimizer')}")
        config_dict.pop("optimizer")
        for k, v in config_dict.items():
            v = str(v).lower() if isinstance(v, bool) else v
            overrides.append(f"optimizer.{k}={v}")
    if loss_config:
        import json
        config_dict: dict = json.loads(loss_config)
        overrides.append(f"loss={config_dict.get('loss')}")
        config_dict.pop("loss")
        for k, v in config_dict.items():
            v = str(v).lower() if isinstance(v, bool) else v
            overrides.append(f"loss.{k}={v}")
    if trainer_config:
        import json
        config_dict: dict = json.loads(trainer_config)
        overrides.append(f"trainer={config_dict.get('trainer')}")
        config_dict.pop("trainer")
        for k, v in config_dict.items():
            v = str(v).lower() if isinstance(v, bool) else v
            overrides.append(f"trainer.{k}={v}")
    
    if overrides:
        command.extend(overrides)
    
    return ContainerSpec(
        image="unet:v0.1.0",
        command=command,
    )


@pipeline
def pipeline(
    # Training parameters
    train_dataset_config: Optional[dict] = None,
    train_network_config: Optional[dict] = None,
    train_optimizer_config: Optional[dict] = None,
    train_loss_config: Optional[dict] = None,
    train_trainer_config: Optional[dict] = None,
):
    train_task = train(
        dataset_config=train_dataset_config,
        network_config=train_network_config,
        optimizer_config=train_optimizer_config,
        loss_config=train_loss_config,
        trainer_config=train_trainer_config,
    )
    
 
if __name__ == "__main__":
    from kfp.compiler import Compiler
    
    Compiler().compile(pipeline, "pipeline.yaml")
