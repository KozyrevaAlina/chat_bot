from torch.cuda import is_available
name_model = 'models/bi_encoder' 
name_tokenizer = 'models/distilbert-base-uncased'
name_cross_encoder = 'models/cross_encoder' 
response_embeddings = 'utils/response_embeddings.npy'
responses = 'utils/responses.csv'

device = "cuda" if is_available() else "cpu"

TOKEN = "6990006031:AAETfBEaNC08VigECLNlDcKMWVxY_MrERB4"

MSG = """I'm Sheldon Cooper, the character from The Big Bang Theory. My IQ = 187 and I have an eidetic memory. I am interested in physics, especially quantum physics, my scientific work is devoted to String Theory. Let's talk"""
