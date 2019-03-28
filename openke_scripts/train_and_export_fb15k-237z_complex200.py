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
con.set_dimension(200)
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

embeddings = con.get_parameters()

complex_name_map = {
    'ent_re_embeddings.weight': 'entities_r.p',
    'ent_im_embeddings.weight': 'entities_i.p',
    'rel_re_embeddings.weight': 'relations_r.p',
    'rel_im_embeddings.weight': 'relations_i.p'
}

transe_name_map = {
    'ent_embeddings.weight': 'entities.p',
    'rel_embeddings.weight': 'relations.p'
}

def save_embedding(embedding, filename) :
    with open(filename, "wb") as f:
        pickle.dump(embedding.numpy(), f)

for emb_name, filename in complex_name_map.items():
    print("Saving to %s" % (directory + filename))
    save_embedding(embeddings[emb_name], directory + filename) 

print("Will execute test set evaluation after ipython shell")

con.test()

