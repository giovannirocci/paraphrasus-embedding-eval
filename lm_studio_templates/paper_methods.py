from lm_studio_templates.templates import predict_lm_studio, local_client

def predict_llama3_p1(batch):
    model = "meta-llama-3-8b-instruct"
    return predict_lm_studio(client=local_client, model=model, batch=batch, p=1, icl=False)


def predict_llama3_p2(batch):
    model = "meta-llama-3-8b-instruct"
    return predict_lm_studio(client=local_client, model=model, batch=batch, p=2, icl=False)


def predict_llama3_p3(batch):
    model = "meta-llama-3-8b-instruct"
    return predict_lm_studio(client=local_client, model=model, batch=batch, p=3, icl=False)

def predict_llama3_iclk4_p1(batch):
    model = "meta-llama-3-8b-instruct"
    return predict_lm_studio(client=local_client, model=model, batch=batch, p=1, icl=True, icl_alt=False)


def predict_llama3_iclk4_p2(batch):
    model = "meta-llama-3-8b-instruct"
    return predict_lm_studio(client=local_client, model=model, batch=batch, p=2, icl=True, icl_alt=False)

def predict_llama3_iclk4_p3(batch):
    model = "meta-llama-3-8b-instruct"
    return predict_lm_studio(client=local_client, model=model, batch=batch, p=3, icl=True, icl_alt=False)