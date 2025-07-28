
export HF_ENDPOINT='https://hf-mirror.com'
export HF_DATASETS_CACHE=/data2/zzd/cache
export HF_HOME=/data2/zzd/cache

save_path=/data2/zzd/data/search_r1
index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --faiss_gpu