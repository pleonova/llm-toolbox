import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class HuggingFaceTextEmbedder:
    """
    A text embedding generator using Hugging Face models.
    """
    def __init__(self, hf_model_id):
        """
        Initializes the HuggingFaceTextEmbedder.

        Args:
            hf_model_id (str): The Hugging Face model identifier.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        self.model = AutoModel.from_pretrained(hf_model_id)

    def _mean_pooling(self, model_output, attention_mask):
        """
        Applies mean pooling to the model's output embeddings.

        Args:
            model_output: The output of the Hugging Face model.
            attention_mask: The attention mask for input tokens.

        Returns:
            torch.Tensor: The mean-pooled embeddings.
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def generate_sentence_embeddings(self, text_list):
        """
        Generates text embeddings for a list of texts using the Hugging Face model.

        Args:
            text_list (list): List of input texts.

        Returns:
            numpy.ndarray: An array containing the generated text embeddings (converted torch.Tensor).

        """
        encoded_input = self.tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings = sentence_embeddings.cpu().detach().numpy()

        return sentence_embeddings

    def generate_text_embeddings_dictionary(self, text_list):
        """
        Generates text embeddings for a list of texts using the selected text embedding generator.

        Args:
            model_text_embedder (class instance): 
            text_list (list): List of input texts.

        Returns:
            dict: A dictionary mapping input texts to their corresponding embeddings.
        """
        embeddings_dict = {}

        embeddings = self.generate_sentence_embeddings(text_list)

        # Map each text to its corresponding embedding
        for t, text in enumerate(text_list):
            embeddings_dict[text] = [embeddings[t]]

        return embeddings_dict
