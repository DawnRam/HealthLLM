import sys
import json
import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from handler import DataHandler

assert torch.cuda.is_available(), "No cuda device detected"


class Inferer:
    """
    A basic inference class for accessing medAlpaca models programmatically.

    This class provides methods for loading supported medAlpaca models, tokenizing inputs,
    and generating outputs based on the specified model and configurations.

    Attributes:
        available_models (dict): A dictionary containing the supported models and their configurations.

    Args:
        model_name (str): The name of the medAlpaca model to use for inference.
        prompt_template (str): The path to the JSON file containing the prompt template.
        base_model (str, optional): If LoRA is used, this should point to the bases model weigts
        model_max_length: (int, optional): Number of input tokens to the model. Default is 512.
        load_in_8bit (bool, optional): Wether a quantized model should be loaded. Default is False
        torch_dtype (torch.dtype, optional): The torch datatype to load the base model. Default is float16
        peft (bool, optional): If the model was trainied in 8bit or with LoRA, PEFT library should be used
            to load the model. Default is False. 

    Example:

        medalpaca = medAlapaca("medalpaca/medalapca-7b", "prompts/alpaca.json")
        response = medalpaca(input="What is Amoxicillin?")
    """
        
    def __init__(
        self, 
        model,
        tokenizer,
        prompt_template: str,
        base_model: str = None,
        model_max_length: int = 512,
        load_in_8bit: bool = False, 
        torch_dtype: torch.dtype = torch.float16, 
        peft: bool = False
    ) -> None:
        
        self.model = model
        self.tokenizer = tokenizer

        self.data_handler = DataHandler(
            tokenizer,
            prompt_template = prompt_template, 
            model_max_length = model_max_length,
            train_on_inputs = False,
        )
        
        
    def __call__(
        self,
        input: str,
        instruction: str = None,
        output: str = None,
        max_new_tokens: int = 128,
        verbose: bool = False,
        **generation_kwargs,
    ) -> str:
        """
        Generate a response from the medAlpaca model using the given input and instruction.

        Args:
            input (str):
                The input text to provide to the model.
            instruction (str, optional):
                An optional instruction to guide the model's response.
            output (str, optional): 
                Prepended to the models output, e.g. for 1-shot prompting
            max_new_tokens (int, optional): 
                How many new tokens the model can generate
            verbose (bool, optional): 
                If True, print the prompt before generating a response.
            **generation_kwargs:
                Keyword arguments to passed to the `GenerationConfig`.
                See here for possible arguments: https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/text_generation

        Returns:
            str: The generated response from the medAlpaca model.
        """
        self.model.eval()
        prompt = self.data_handler.generate_prompt(instruction = instruction, input = input, output = output)
        if verbose:
            print(prompt)
            
        input_tokens = self.data_handler.tokenizer(prompt, return_tensors="pt")
        input_token_ids = input_tokens["input_ids"].to("cuda")

        generation_config = GenerationConfig(**generation_kwargs)

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_token_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        generation_output_decoded = self.data_handler.tokenizer.decode(generation_output.sequences[0])
        split = f'{self.data_handler.prompt_template["output"]}{output or ""}'
        response = generation_output_decoded.split(split)[-1].strip()
        return response
