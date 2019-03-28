from openke import config, models
import pickle
import pathlib


con = config.Config()
con.set_in_path("./data/dbpedia50/openke_format/")  # please update paths

# True: Input test files from the same folder.
con.set_test_link_prediction(True)

con.set_log_on(1)
con.set_work_threads(25)
con.set_train_times(900)
con.set_nbatches(100)
con.set_alpha(0.003)
con.set_bern(1)
con.set_dimension(300)
con.set_margin(1.0)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")


# Model parameters will be exported via torch.save() automatically.
# Not needed for us. Just write to /tmp and forget, because OpenKE will crash if nothing is passed.
con.set_export_files("/tmp/complex.pt")


#Model parameters will be exported to json files automatically.
# Not needed for us. Just write to /tmp and forget, because OpenKE will crash if nothing is passed.
con.set_out_files("/tmp/complex.vec.json")

con.init()
con.set_model(models.ComplEx)
con.run()


####### Export ########
embeddings = con.get_parameters()
directory = 'models/dbpedia50/complex300/'
pathlib.Path(directory).mkdir(exist_ok=True, parents=True)

complex_name_map = {
    'ent_re_embeddings.weight': 'entities_r.p',
    'ent_im_embeddings.weight': 'entities_i.p',
    'rel_re_embeddings.weight': 'relations_r.p',
    'rel_im_embeddings.weight': 'relations_i.p'
}

other_name_map = {
    'ent_embeddings.weight': 'entities.p',
    'rel_embeddings.weight': 'relations.p'
}

def save_torch_embedding_as_numpy(embedding, filename):
    with open(filename, "wb") as f:
        pickle.dump(embedding.numpy(), f)

for emb_name, filename in complex_name_map.items():
    print("Saving to %s" % (directory + filename))
    save_torch_embedding_as_numpy(embeddings[emb_name], directory + filename)


print("Will execute test set evaluation after ipython shell")
con.test()

