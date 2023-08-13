import importlib
from omegaconf import OmegaConf


def get_class(cls_path):
    package_path = '.'.join(cls_path.split('.')[:-1])
    cls_name = cls_path.split('.')[-1]
    package = importlib.import_module(package_path)
    return getattr(package, cls_name)


def export(cls_path: str, model_name: str):
    cls = get_class(cls_path)
    model = cls.from_pretrained(model_name)
    model.export(f'{model_name}.onnx')
    print(OmegaConf.to_yaml(model._cfg))


cls_path = 'nemo.collections.asr.models.EncDecClassificationModel'
cls_path = 'nemo.collections.asr.models.EncDecCTCModel'
cls_path = 'nemo.collections.asr.models.EncDecClassificationModel'
model_name = 'vad_marblenet'
model_name = 'stt_en_quartznet15x5'
model_name = 'stt_en_jasper10x5dr'
model_name = 'commandrecognition_en_matchboxnet3x1x64_v2'
export(cls_path, model_name)
