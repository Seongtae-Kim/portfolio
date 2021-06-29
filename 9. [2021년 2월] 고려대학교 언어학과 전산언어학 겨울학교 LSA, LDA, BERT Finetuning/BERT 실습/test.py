from winterschool_trainer import Trainer
model = "/home/seongtae/SynologyDrive/SIRE/Projects/KR-BERT/KR-BERT/krbert_pytorch/pretrained/pytorch_model_char16424_ranked.bin"
config = "/home/seongtae/SynologyDrive/SIRE/Projects/KR-BERT/KR-BERT/krbert_pytorch/pretrained/bert_config_char16424.json"
tokenizer = '/home/seongtae/SynologyDrive/SIRE/Projects/KR-BERT/KR-BERT/krbert_pytorch/pretrained/vocab_snu_char16424.txt'

t = Trainer(model_path=model, config_path=config,
            tokenizer_path=tokenizer, train_ratio=0.9, batch_size=8, epoch=1)
nsmc = t.build_corpus("nsmc")
texts = nsmc.get_all_texts()
labels = nsmc.get_all_labels()

t.encode(texts[:100], labels[:100])
t.prepare()
t.train()
