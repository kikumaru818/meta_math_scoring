from models import finetune

REGISTERED_MODELS = {
    'base': finetune.BaseModel,
    'proto': finetune.ProtoModel,
}