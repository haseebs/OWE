import pathlib
import pickle

import config
import models


con = config.Config()
con.set_work_threads(8)

# Dataset
con.set_in_path("../data/FB15k-237-zeroshot/openke_format/")

# Model Parameter
con.set_nbatches(100)
con.set_train_times(400)
con.set_alpha(0.03)
con.set_bern(1)
con.set_dimension(300)
con.set_margin(1.0)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")

# Testing
con.set_valid_steps(10)
# con.set_early_stopping_patience(10)
con.set_checkpoint_dir("./checkpoint")
con.set_result_dir("./result")
con.set_save_steps(10)
con.set_test_link(True)
con.set_test_triple(True)

# Start training
con.init()
# con.set_train_model(models.ComplEx)
con.set_test_model(models.ComplEx)
# con.train()


######################################################
# Export
######################################################

model = con.trainModel or con.testModel

complex_name_map = {
    'ent_re_embeddings': 'entities_r.p',
    'ent_im_embeddings': 'entities_i.p',
    'rel_re_embeddings': 'relations_r.p',
    'rel_im_embeddings': 'relations_i.p'
}

other_name_map = {
    'ent_embeddings': 'entities.p',
    'rel_embeddings': 'relations.p'
}

def save_embedding(embedding, filename) :
    with open(filename, "wb") as f:
        try:
            pickle.dump(embedding.weight.cpu().numpy(), f)
        except RuntimeError:
            pickle.dump(embedding.weight.detach().cpu().numpy(), f)

out_directory = "../pretrained_models/fb15k-237-owe/complex300/"
pathlib.Path(out_directory).mkdir(exist_ok=True, parents=True)

for emb_name, filename in complex_name_map.items():
    print("Saving to %s" % (out_directory + filename))
    save_embedding(getattr(model, emb_name), out_directory + filename)

con.test()


