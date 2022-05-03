from sentence_transformers import SentenceTransformer
from transformers.modeling_outputs import  SequenceClassifierOutput
import torch
from torch import nn
from torch.nn import CrossEntropyLoss



class SBERT_Classifer(nn.Module):
    def __init__(self,args=None):
        super().__init__()
        self.args = args
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.embed_output = self.model.get_sentence_embedding_dimension()
        self.n_labels = 5
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.embed_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_labels),
        )

        self.linear_relu_stack.apply(self.weights_init_uniform)

    def weights_init_uniform(self,m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,):

        inputs = self.model.encode(input_ids) #(bsz, embe_out)
        inputs = torch.Tensor(inputs)
        inputs = inputs.to(self.model.device)
        logits = self.linear_relu_stack(inputs)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

