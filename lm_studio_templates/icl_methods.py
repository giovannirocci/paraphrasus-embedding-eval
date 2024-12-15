from lm_studio_samples.samples import predict_lm_studio, local_client, remote_client

def predict_llama3_iclk4_p1_alt(batch):
    model = "meta-llama-3-8b-instruct"
    return predict_lm_studio(client=local_client, model=model, batch=batch, p=1, icl=True, icl_alt=True)

def predict_llama3_iclk4_p2_alt_remote(batch):
    model = "meta-llama-3-8b-instruct"
    return predict_lm_studio(client=remote_client, model=model, batch=batch, p=2, icl=True, icl_alt=True)

def predict_llama3_iclk4_p2_remote(batch):
    model = "meta-llama-3-8b-instruct"
    return predict_lm_studio(client=remote_client, model=model, batch=batch, p=2, icl=True, icl_alt=False)
