import torch
import torch.nn.functional as F

# Example logits for the first token
logits_1 = torch.tensor([2.0, 1.0, 0.1])  # Shape [vocab_size]

# Example logits for the second token given the first token
# Assume this is a matrix where each row corresponds to logits for the second token
# conditioned on one of the first token's possibilities
logits_2 = torch.tensor([[0.5, 1.5], [0.2, 1.0], [0.8, 1.2]])  # Shape [vocab_size, vocab_size]

# Temperature parameter
temperature = 0.7

# 1. Scale logits by temperature
logits_1_scaled = logits_1 / temperature  # Shape [vocab_size]
logits_2_scaled = logits_2 / temperature  # Shape [vocab_size, vocab_size]

# 2. Convert logits to probabilities using softmax
probs_1 = F.softmax(logits_1_scaled, dim=-1)  # Shape [vocab_size]
probs_2 = F.softmax(logits_2_scaled, dim=-1)  # Shape [vocab_size, vocab_size]

# 3. Compute joint probability distribution for the pair of tokens
joint_probs = probs_1.unsqueeze(1) * probs_2  # Shape [vocab_size, vocab_size]

# 4. Marginalize out the second token (i.e., sum over the second token's dimension)
marginal_probs_1 = joint_probs.sum(dim=1)  # Shape [vocab_size]

# 5. Convert marginal probabilities back to logits if needed
marginal_logits_1 = torch.log(marginal_probs_1)  # Shape [vocab_size]

print("Logits for the first token after marginalizing out the second token:")
print(marginal_logits_1)
