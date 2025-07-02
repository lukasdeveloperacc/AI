from abc import ABC, abstractmethod
from common.utils import DatasetUtil

import pandas as pd
import torch, os, logging


class BaseExperimentLogger(ABC):
    def __init__(self, experiment_name: str, experiment_dir: str):
        self._experiment_name = experiment_name
        self._experiment_dir = experiment_dir

    @abstractmethod
    def log_param(self, key, value):
        pass

    @abstractmethod
    def log_metric(self, key, value, step=None):
        pass

    @abstractmethod
    def log_artifact(self, path: str):
        pass

    @abstractmethod
    def log_model(self, model, is_state_dict: bool):
        pass

    @abstractmethod
    def end(self):
        pass


class BaseDataset(ABC, torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        data_extensions: list[str],
        data_frame: pd.DataFrame | None = None,
        transform: torch.nn.Module | None = None,
        target_mask_transform: torch.nn.Module | None = None,
        target_data_class_transform: torch.nn.Module | None = None,
    ):
        self._classes: tuple | None = None
        self._root_dir = root_dir
        self._data_extensions = data_extensions

        if transform is None:
            self._transform = self.make_transform()
        else:
            self._transform = transform

        if target_mask_transform is None:
            self._target_mask_transform = self.make_transform_mask()
        else:
            self._target_mask_transform = target_mask_transform

        if target_data_class_transform is None:
            self._target_data_class_transform = self.make_transform_data_class()
        else:
            self._target_data_class_transform = target_data_class_transform

        self._train_df, self._valid_df, self._test_df = (None, None, None)

        if data_frame is None:
            self._df = pd.DataFrame(columns=["image_path", "mask_path", "class"])
            self._df = self.make_data_frame()
            self._train_df, self._test_df = DatasetUtil.train_test_split_from_df(
                self._df, test_size=0.2, random_state=42
            )
            self._train_df, self._valid_df = DatasetUtil.train_test_split_from_df(
                self._train_df, test_size=0.1, random_state=42
            )
            self._classes = list(set(self._df["class"]))
            logging.info(f"Class nums : {len(self._classes)}")
        else:
            self._df = data_frame

        logging.info(f"Number of samples: {len(self._df)}")

    @property
    def classes(self) -> list[str]:
        return self._classes

    @classes.setter
    def classes(self, classes: list[str]) -> None:
        self._classes = classes

    @abstractmethod
    def download(self):
        pass

    @abstractmethod
    def make_data_frame(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def make_transform(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def make_transform_data_class(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def make_transform_mask(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def __getitem__(self, idx) -> tuple:
        pass

    def __len__(self) -> int:
        return len(self._df)

    def _load_dataset(self, phase: str, batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
        if phase.lower() == "train":
            df = self._train_df
        elif phase.lower() == "valid":
            df = self._valid_df
        else:
            df = self._test_df

        logging.info(f"Loaded {phase} dataset")
        dataset = self.__class__(
            root_dir=self._root_dir,
            data_extensions=self._data_extensions,
            data_frame=df,
            transform=self._transform,
            target_mask_transform=self._target_mask_transform,
            target_label_transform=self._target_data_class_transform,
        )

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return data_loader

    def load_train_dataset(self, batch_size: int = 32, shuffle: bool = True) -> torch.utils.data.DataLoader:
        return self._load_dataset(phase="train", batch_size=batch_size, shuffle=shuffle)

    def load_valid_dataset(self, batch_size: int = 32, shuffle: bool = True) -> torch.utils.data.DataLoader:
        return self._load_dataset(phase="valid", batch_size=batch_size, shuffle=shuffle)

    def load_test_dataset(self, batch_size: int = 32, shuffle: bool = True) -> torch.utils.data.DataLoader:
        return self._load_dataset(phase="test", batch_size=batch_size, shuffle=shuffle)


class BaseTrainer(ABC):
    def __init__(
        self,
        dataset: BaseDataset | list[BaseDataset],
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epochs: int,
        batch_size: int,
        including_test: bool = True,
        scheduler=None,
        logger: BaseExperimentLogger | None = None,
        device: str = "cpu",
        checkpoint_dir: str = "outputs/checkpoints",
        pretrained_checkpoint_path: str = "",
        is_init_vram: bool = True,
        is_export_onnx: bool = False,
        is_export_tensorrt: bool = False,
    ) -> None:
        self._epochs = epochs
        self._batch_size = batch_size
        self._including_test = including_test
        self._device = device
        logging.info(f"Device : {self._device}")

        self._is_export_onnx = is_export_onnx
        self._is_export_tensorrt = is_export_tensorrt

        if is_init_vram:
            self.initialize_gpu_memory()

        # TODO: Where is it processed
        if isinstance(dataset, list):
            torch.utils.data.ConcatDataset(dataset)
        else:
            self._dataset = dataset

        self._experiment_logger = logger

        self._network = network.to(self._device)
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._scheduler = scheduler
        self._checkpoint_dir = checkpoint_dir
        if self._checkpoint_dir:
            os.makedirs(self._checkpoint_dir, exist_ok=True)

        self._pretrained_checkpoint_path = pretrained_checkpoint_path
        if self._pretrained_checkpoint_path:
            self.load_checkpoint(self._pretrained_checkpoint_path)

        self._example_input: torch.Tensor | None = None

    def train(self, start_epoch: int = 0) -> None:
        try:
            for epoch in range(start_epoch + 1, self._epochs + 1):
                self._network.train()
                train_log_dict: dict = self.train_one_epoch(epoch)
                for key, value in train_log_dict.items():
                    self._experiment_logger.log_metric(key=key, value=value, step=epoch)

                self._pretrained_checkpoint_path = self.save_checkpoint(epoch)

                self._network.eval()
                val_log_dict = self.validate(epoch)
                for key, value in val_log_dict.items():
                    self._experiment_logger.log_metric(key=key, value=value, step=epoch)

        except Exception as e:
            logging.error(e)

        finally:

            if self._including_test:
                self._network.eval()
                test_log_dict = self.test()
                for key, value in test_log_dict.items():
                    self._experiment_logger.log_metric(key=key, value=value, step=epoch)

            self._network.train()
            self._experiment_logger.log_model(self._network)

            self._experiment_logger.end()

    @abstractmethod
    def train_one_epoch(self) -> dict:
        pass

    @abstractmethod
    def validate(self, epoch: int) -> dict:
        pass

    @abstractmethod
    def test(self) -> dict:
        pass

    def save_checkpoint(self, epoch: int | None = None) -> str:
        checkpoint_path = os.path.join(self._checkpoint_dir, f"epoch_{epoch}.pt")
        torch.save(self._network.state_dict(), checkpoint_path)
        logging.info(f"Save to {checkpoint_path}")

        return checkpoint_path

    def export_onnx(self) -> None:
        export_path = os.path.join(self._checkpoint_dir, f"model.onnx")

        if self._example_input is None:
            raise AttributeError(f"self._example_input is not set")
        else:
            self._example_input = self._example_input.detach()
            self._example_input.requires_grad = False

        self._network.eval()
        torch.onnx.export(
            self._network,
            (self._example_input,),
            export_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        logging.info(f"Export onnx to {export_path}")

        import onnx

        model = onnx.load(export_path)
        onnx.checker.check_model(model)
        logging.info(f"onnx model input name : {model.graph.input[0].name}")
        logging.info(f"onnx model input type : {model.graph.input[0].type}")
        output = model.graph.output[0]
        logging.info(f"onnx model output name : {output.name}")
        logging.info(f"onnx model output shape : {output.type.tensor_type.shape}")

        import onnxruntime
        import numpy as np

        ort_session = onnxruntime.InferenceSession(
            export_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        logging.info(f"onnx runtime session providers : {ort_session.get_providers()}")

        def to_numpy(tensor: torch.Tensor):
            return tensor.cpu().numpy()

        # ONNX 런타임에서 계산된 결과값
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(self._example_input)}
        ort_outs = ort_session.run(None, ort_inputs)
        ort_out = ort_outs[0]
        logging.info(f"ort output shape : {ort_out.shape}")

        with torch.no_grad():
            torch_out: torch.Tensor = self._network(self._example_input)
            logging.info(f"torch output shape : {torch_out.shape}")

        try:
            np.testing.assert_allclose(to_numpy(torch_out), ort_out, rtol=1e-03, atol=1e-05)
            logging.info("Exported model has been tested with ONNXRuntime")

        except Exception as e:
            logging.error(e)

    def export_tensorrt(self) -> None:
        target_onnx_path = os.path.join(self._checkpoint_dir, f"model.onnx")
        trt_engine_path = os.path.join(self._checkpoint_dir, f"model.plan")
        import tensorrt as trt
        import numpy as np
        import pycuda.driver as cuda
        import pycuda.autoinit  # CUDA context 자동 초기화

        logging.info(f"TensorRT version : {trt.__version__}")

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        engine = None
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:

            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

            with open(target_onnx_path, "rb") as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            example_shape = tuple(self._example_input.shape)
            profile = builder.create_optimization_profile()
            input_name = network.get_input(0).name
            profile.set_shape(
                input_name, min=(1,) + example_shape[1:], opt=(4,) + example_shape[1:], max=(8,) + example_shape[1:]
            )
            config.add_optimization_profile(profile)

            engine = builder.build_serialized_network(network, config)
            with open(trt_engine_path, "wb") as f:
                f.write(engine)

        logging.info(f"Saved to {trt_engine_path}")

        logging.info(f"Start TensorRT inference for testing")

        with open(trt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()

        input_shape = (1,) + example_shape[1:]
        input_data = np.random.rand(*input_shape).astype(np.float32)

        input_name = engine.get_tensor_name(0)
        output_name = engine.get_tensor_name(1)
        context.set_input_shape(input_name, input_data.shape)

        d_input = cuda.mem_alloc(int(input_data.nbytes))

        output_shape = context.get_tensor_shape(output_name)
        d_output = cuda.mem_alloc(int(np.prod(output_shape) * input_data.itemsize))

        cuda.memcpy_htod(d_input, input_data)

        bindings = [int(d_input), int(d_output)]
        context.execute_v2(bindings)

        output_data = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output_data, d_output)

        logging.info(f"TensorRT inference result shape : {output_data.shape}")

    def load_checkpoint(self, path: str) -> None:
        self._network.load_state_dict(torch.load(path))
        logging.info(f"Loaded state dict from {path}")

    def log(self, epoch: int | None = None, log_dict: dict | None = None) -> None:
        if log_dict:
            for key, value in log_dict.items():
                self._experiment_logger.log_metric(key=key, value=value, step=epoch)

            logging.info(f"[Epoch : {epoch}] : {log_dict}")

    @staticmethod
    def initialize_gpu_memory():
        torch.cuda.empty_cache()
        logging.info("Empty cache")
        torch.cuda.reset_peak_memory_stats()
        logging.info("Reset peak memory stats")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
