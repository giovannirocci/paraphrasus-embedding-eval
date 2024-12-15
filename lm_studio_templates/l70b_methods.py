from lm_studio_templates.templates import predict_lm_studio, local_client

def predict_llama3_3_70b_ins_8bit(batch):
    model = "llama-3.3-70b-instruct"

    return predict_lm_studio(client=local_client, model=model, batch=batch, p=1)