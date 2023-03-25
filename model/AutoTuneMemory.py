import torch
import torch.nn as nn

# LLM auto memory module
# Provide a mechanism for self pruning of the memory context.
#
# Score Net:
# - Responsible for computing the importance scores for both the new sentence
#   tokens and the current memory context.
#
# Threshold Net:
# - Calculates a dynamic threshold factor based on the mean of the memory context.
#   The threshold is then used to filter the combined tokens, retaining only the ones
#   above the threshold.

# The memory context is updated with the filtered tokens and their corresponding
# importance scores.
#
# If the memory context exceeds the maximum size, it is trimmed down to the top most
# important tokens.

class AutoMemoryModule(nn.Module):
    def __init__(self, token_dim, hidden_dim, max_memory_size):
        super(AutoMemoryModule, self).__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.max_memory_size = max_memory_size

        self.importance = torch.randn(0)

        self.score_net = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.threshold_net = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, sentence_tokens, memory_context):
        # Compute importance scores for the new tokens and the current memory context
        new_importance_scores = self.score_net(sentence_tokens).squeeze()
        current_importance_scores = self.score_net(memory_context).squeeze()

        # Calculate the dynamic threshold for retaining tokens
        context_mean = memory_context.mean(dim=0)
        threshold_factor = self.threshold_net(context_mean).item()

        # Combine new tokens and current memory context, along with their importance scores
        combined_tokens = torch.cat([memory_context, sentence_tokens], dim=0)
        combined_importance = torch.cat([current_importance_scores, new_importance_scores], dim=0)

        # Filter the tokens based on the dynamic threshold
        threshold = threshold_factor * combined_importance.max()
        mask = combined_importance >= threshold

        # Update the memory context and importance
        memory_context = combined_tokens[mask]
        self.importance = combined_importance[mask]

        # Limit the size of the memory context if it exceeds the maximum size
        if memory_context.size(0) > self.max_memory_size:
            sorted_indices = torch.argsort(self.importance, descending=True)
            memory_context = memory_context[sorted_indices[:self.max_memory_size]]
            self.importance = self.importance[sorted_indices[:self.max_memory_size]]

        return memory_context.clone().detach(), combined_importance.clone().detach()
