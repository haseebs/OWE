from openke import config, models
import json
import pickle

import pathlib


directory = "/amigo/LAVIS/johannes/res/fb15k-237z/transe/"
pathlib.Path(directory).mkdir(exist_ok=True, parents=True)

con = config.Config()
#Input training files from benchmarks/FB15K/ folder.
con.set_in_path("./benchmarks/FB15k-237-zeroshot/with_id/")
#True: Input test files from the same folder.
print("setting test true")
con.set_test_link_prediction(True)
print("setting params")
con.set_log_on(1)
con.set_work_threads(8)
con.set_train_times(1000)
con.set_nbatches(100)
con.set_alpha(0.1)
con.set_bern(1)
con.set_dimension(300)
con.set_margin(1.0)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")
#Model parameters will be exported via torch.save() automatically.
con.set_export_files("/tmp/johannes_villmow/transe.pt")
#Model parameters will be exported to json files automatically.
con.set_out_files("/tmp/johannes_villmow/transe.vec.json")
con.init()
con.set_model(models.TransE)
con.run()

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

for emb_name, filename in transe_name_map.items():
    print("Saving to %s" % (directory + filename))
    save_embedding(embeddings[emb_name], directory + filename) 

con.test()
