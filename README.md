# Speaker_verification
Speaker Verification Project with Speechbrain

## Speechbrain Model

- **Model:** EncoderClassifier
    
    [https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
    
- **Installation**
    
    ```bash
    conda create -n speechbrain python=3.9 pip
    conda activate speechbrain
    
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    #pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0
    pip install ffmpeg
    
    pip install speechbrain
    
    -----------------------------설치 완료------------------------------
    
    # torchaudio에서 library error 나면 실행 (MAC OS기준)
    git clone https://github.com/pytorch/audio.git
    cd audio
    CC=clang CXX=clang++ pythpip ion setup.py install
    ```
    
- **Example**

    ```python
    import torchaudio
    from speechbrain.pretrained import EncoderClassifier
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
    signal, fs =torchaudio.load('tests/samples/ASR/spk1_snt1.wav')
    embeddings = classifier.encode_batch(signal)
    #embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
    
    # the resulting embeddings can be used for cosine similarity-based retrieval
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    similarity = cosine_sim(embeddings[0], embeddings[1])
    threshold = 0.5  # the optimal threshold is dataset-dependent
    if similarity < threshold:
        print("Speakers are not the same!")
    round(similarity.item(), 2)
    ```
    
## Train with Custom Datasets
; Speechbrain에서 제공하는 templates에 있는 speaker_id 모델과 recipes에 있는 Voxceleb 데이터를 이용한 speaker embeddings 모델에 있는 train, data_prepare, hyperprameter 코드를 참고하여 우리 데이터에 맞게 Train 코드 구현

- **Train**

    ```bash
    python train.py train.yaml
    ```

- **Data Parallel Train**

    ```bash
    CUDA_VISIBLE_DEVICES=0,1 python train.py train.yaml --data_parallel_backend
    ```

- **Distributed Data Parallel Train**

    ```bash
    python -m torch.distributed.launch --nproc_per_node=4 train.py train.yaml --distributed_launch --distributed_backend='nccl'
    ```
    https://speechbrain.readthedocs.io/en/latest/multigpu.html
