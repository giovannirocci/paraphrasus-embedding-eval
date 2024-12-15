from lm_studio_samples.samples import predict_lm_studio, client

def predict_llama3_3_70b_ins_8bit(batch):
    model = "llama-3.3-70b-instruct"

    return predict_lm_studio(client=client, model=model, batch=batch, p=1)