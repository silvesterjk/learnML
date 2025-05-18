from torch import nn
import torch
from ..network import SCORE_SAVER

class ScaledDotAttention(nn.Module):

    def __init__(self,
                 d_k,
                 dropout: float = 0.0):
        """

        Args:
            d_k: Dimension of Keys and Queries
            dropout: Dropout probability
        """
        super().__init__()
        self.d_k = d_k

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = None

        ########################################################################
        # TODO:                                                                #
        #   Task 11: Initialize the dropout layer (torch.nn implementation)    #
        #                                                                      #           
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the scaled dot attention given query, key and value inputs. Stores the scores in SCORE_SAVER for
        visualization

        Args:
            q: Query Inputs
            k: Key Inputs
            v: Value Inputs
            mask: Optional Causal or Padding Boolean Mask

        Shape:
            - q: (*, sequence_length_queries, d_model)
            - k: (*, sequence_length_keys, d_model)
            - v: (*, sequence_length_keys, d_model)
            - mask: (*, sequence_length_queries, sequence_length_keys)
            - outputs: (*, sequence_length_queries, d_v)
        """

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        ########################################################################
        # TODO:                                                                #
        #   Task 6:                                                            #
        #       - Add a negative infinity mask if a mask is given              #
        #                                                                      #    
        # Hint 6:                                                              #
        #       - Have a look at Tensor.masked_fill_() or use torch.where()    #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        scores = self.softmax(scores)

        ########################################################################
        # TODO:                                                                #
        #   Task 11:                                                           #
        #       - Add dropout to the scores                                    #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        
        outputs = torch.matmul(scores, v)

        SCORE_SAVER.save(scores)

        return outputs
