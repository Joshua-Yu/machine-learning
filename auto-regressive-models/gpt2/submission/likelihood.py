import torch
import torch.nn as nn

def log_likelihood(model, text):
    """
    Compute the log-likelihoods for a string `text`
    :param model: The GPT-2 model
    :param texts: A tensor of shape (1, T), where T is the length of the text
    :return: The log-likelihood. It should be a Python scalar.
        NOTE: for simplicity, you can ignore the likelihood of the first token in `text`.
    """

    with torch.no_grad():
        ## TODO:
        ##  1) Compute the logits from `model`;
        ##  2) Return the log-likelihood of the `text` string. It should be a Python scalar.
        ##      NOTE: for simplicity, you can ignore the likelihood of the first token in `text`
        ##      Hint: Checkout Pytorch softmax: https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
        ##                     Pytorch negative log-likelihood: https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
        ##                     Pytorch Cross-Entropy Loss: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        ## 
        ## The problem asks us for (positive) log likelihood, which would be equivalent to the negative of a negative log likelihood. 
        ## Cross Entropy Loss is equivalent applying LogSoftmax on an input, followed by NLLLoss. Use reduction 'sum'.
        ## 
        ## Hint: Implementation should only takes 3~7 lines of code.
        
        ### START CODE HERE ###

        # compute logits using model

        logits, _ = model(text)

        # Shift targets so they align with the logits (predicting the next token)
        targets = text[:, 1:]  # Remove the first token for target alignment
        logits = logits[:, :-1, :]  # Remove the last logits to match the target size

        # Reshape logits and targets for loss calculation
        logits = logits.reshape(-1, logits.size(-1))  # Flatten logits to shape [batch_size * sequence_length, vocab_size]
        targets = targets.reshape(-1)  # Flatten targets to shape [batch_size * sequence_length]

        # Calculate log-likelihood using CrossEntropyLoss (which internally uses log_softmax + NLLLoss)
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        log_likelihood = -loss_fn(logits, targets)

        # Convert log-likelihood to a Python scalar
        log_likelihood = log_likelihood.item()
        return log_likelihood

        ### END CODE HERE ###
        raise NotImplementedError
