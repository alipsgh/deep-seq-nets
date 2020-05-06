import yaml

from models.char_cnn import CharCNN
from models.fasttext import FastText
from models.attention_rnn import AttentionRNN
from models.rcnn import RCNN
from models.textcnn import TextCNN
from models.textrnn import TextRNN
from models.transformer import Transformer
from utils.logger import get_logger


def instantiate_model(model_name, vocab_size, embeddings):

    if model_name == 'char_cnn':
        model_args = yaml.load(open('./configs/char_cnn.yml'), Loader=yaml.FullLoader)
        model = CharCNN(vocab_size, embeddings, **model_args)

    elif model_name == "rcnn":
        model_args = yaml.load(open('./configs/rcnn.yml'), Loader=yaml.FullLoader)
        model = RCNN(vocab_size, embeddings, **model_args)

    elif model_name == "textcnn":
        model_args = yaml.load(open('./configs/textcnn.yml'), Loader=yaml.FullLoader)
        model = TextCNN(vocab_size, embeddings, **model_args)

    elif model_name == "textrnn":
        model_args = yaml.load(open('./configs/textrnn.yml'), Loader=yaml.FullLoader)
        model = TextRNN(vocab_size, embeddings, **model_args)

    elif model_name == "attention_rnn":
        model_args = yaml.load(open('./configs/attention_rnn.yml'), Loader=yaml.FullLoader)
        model = AttentionRNN(vocab_size, embeddings, **model_args)

    elif model_name == "transformer":
        model_args = yaml.load(open('./configs/transformer.yml'), Loader=yaml.FullLoader)
        model = Transformer(vocab_size, embeddings, **model_args)

    else:
        model_args = yaml.load(open('./configs/fasttext.yml'), Loader=yaml.FullLoader)
        model = FastText(vocab_size, embeddings, **model_args)

    logger = get_logger(__name__)
    logger.info("A model of {} is instantiated.".format(model.__class__.__name__))

    return model

