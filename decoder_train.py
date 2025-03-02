import re
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
# 학습 데이터셋 전처리
def clean_text(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        book_file=f.read()

    cleaned_text=re.sub(r'\n+', ' ', book_file) # 줄바꿈을 빈칸으로 변경
    cleaned_text=re.sub(r'\s+', ' ',cleaned_text) # 여러 빈칸을 하나의 빈칸으로 변경

    with open('cleaned_'+filename, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)


# tiktoken의 gpt2 토크나이저 사용
tokenizer=tiktoken.get_encoding('gpt2')

# 데이터셋 학습할 수 있는 형태로 만들기
class MyDataset(Dataset):
    def __init__(self, txt, max_length, stride):
        self.input_ids=[]
        self.target_ids=[]

        token_ids=tokenizer.encode(txt)

        print(f'# of tokens in txt: {len(token_ids)}')

        for i in range(0,len(token_ids)-max_length,stride):
            input_chunk=token_ids[i:i+max_length]
            target_chunk=token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self,idx):
        return self.input_ids[idx], self.target_ids[idx]

input_file='input.txt'
clean_text(input_file)
with open('cleaned_'+input_file, 'r', encoding='utf-8-sig') as f:
    txt=f.read()

dataset=MyDataset(txt,max_length=32,stride=4)
train_loader=DataLoader(dataset,batch_size=128,shuffle=True,drop_last=True)

VOCAB_SIZE=tokenizer.n_vocab 
CONTEXT_LENGTH=128
EMB_DIM=768
NUM_HEADS=12
NUM_LAYERS=12
DROP_RATE=0.1
QKV_BIAS=False

class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out):
        super().__init__()

        assert d_out%NUM_HEADS==0, "d_out must be devisivle by n_heads"

        self.d_out=d_out
        self.head_dim=d_out//NUM_HEADS

        self.wq=nn.Linear(d_in,d_out,bias=QKV_BIAS)
        self.wk=nn.Linear(d_in,d_out,bias=QKV_BIAS)
        self.wv=nn.Linear(d_in,d_out,bias=QKV_BIAS)
        self.out_proj=nn.Linear(d_out,d_out)
        self.dropout=nn.Dropout(DROP_RATE)
        self.register_buffer('mask',torch.triu(torch.ones(CONTEXT_LENGTH,CONTEXT_LENGTH),diagonal=1))

    def forward(self,x):
        b,num_tokens,d_in=x.shape

        keys=self.wk(x)
        queries=self.wq(x)
        values=self.wv(x)

        keys=keys.view(b,num_tokens,NUM_HEADS,self.head_dim)
        queries=queries.view(b,num_tokens,NUM_HEADS,self.head_dim)
        values=values.view(b,num_tokens,NUM_HEADS,self.head_dim)

        keys=keys.transpose(1,2)
        queries=queries.transpose(1,2)
        values=values.transpose(1,2)

        attn_score=queries @ keys.transpose(2,3)

        mask_bool=self.mask_bool()[:num_tokens,:num_tokens]

        attn_score.masked_fill_(mask_bool,-torch.inf)

        attn_weights=torch.softmax(attn_score/keys.shape[-1]**0.5,dim=-1)
        attn_weights=self.dropout(attn_weights)

        context_vec=(attn_weights@values).transpose(1,2)

        context_vec=context_vec.reshape(b,num_tokens,self.d_out)
        context_vec=self.out_proj(context_vec)

        return context_vec

class LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps=1e-5
        self.scale=nn.Parameter(torch.ones(emb_dim))
        self.shift=nn.Parameter(torch.zeros(emb_dim))

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        var=x.var(dim=-1,keepdim=True,unbiased=False)
        norm_x=(x-mean)/torch.sqrt(var+self.eps)
        return self.scale*norm_x+self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715*torch.pow(x,3))))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(EMB_DIM,4*EMB_DIM),
            GELU(),
            nn.Linear(4*EMB_DIM,EMB_DIM)
        )

    def forward(self,x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn=MultiHeadAttention(d_in=EMB_DIM,d_out=EMB_DIM)
        self.ff=FeedForward()
        self.norm1=LayerNorm(EMB_DIM)
        self.norm2=LayerNorm(EMB_DIM)
        self.drop_shortcut=nn.Dropout(DROP_RATE)

    def forward(self,x):
        shortcut=x
        x=self.norm1(x)
        x=self.attn(x)
        x=self.drop_shortcut(x)
        x=x+shortcut

        shortcut=x
        x=self.norm2(x)
        x=self.ff(x)
        x=self.drop_shortcut(x)
        x=x+shortcut

        return x

class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb=nn.Embedding(VOCAB_SIZE,EMB_DIM)
        self.pos_emb=nn.Embedding(CONTEXT_LENGTH,EMB_DIM)
        self.drop_emb=nn.Dropout(DROP_RATE)

        self.trf_blocks=nn.Sequential(
            *[TransformerBlock() for _ in range(NUM_LAYERS)]
        )

        self.final_norm=LayerNorm(EMB_DIM)
        self.out_head=nn.Linear(EMB_DIM,VOCAB_SIZE,bias=False)

    def forward(self,in_idx):
        batch_size,seq_len=in_idx.shape
        tok_embeds=self.tok_emb(in_idx)
        pos_embeds=self.pos_emb(torch.arange(seq_len,device=in_idx.device))
        x=tok_embeds+pos_embeds
        x=self.trf_blocks(x)
        x=self.final_norm(x)
        logits=self.out_head(x)
        return logits

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=GPTModel()
model.to(deivce)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0.1)

losses=[]
for epoch in range(100):
    model.train()

    epoch_loss=0
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        input_batch,target_batch=input_batch.to(device),target_batch.to(device)

        logits=model(input_batch)
        loss=F.cross_entropy(logits.flatten(0,1), target_batch.flatten())
        epoch_loss+=loss.item()
        loss.backward()
        tokens_seen+=input_batch.numel()
        global_step+=1
        
    avg_loss=epoch_loss/len(train_loader)
    losses.append(avg_loss)
    print(f'Epoch: {epoch+1}, Loss: {avg_loss}')
    torch.save(model.state_dict(), 'model_'+str(epoch+1).zfill(3)+'.pth')

