from app.embedding import embed_texts

# 用两个语义相关、两个语义无关的句子测试
texts = [
    "苹果是一种水果",
    "香蕉是一种水果",
    "汽车是一种交通工具",
]

vectors = embed_texts(texts)

print(f"输入 {len(texts)} 条文本")
print(f"输出 {len(vectors)} 个向量")
print(f"每个向量维度: {len(vectors[0])}")
print()
print("苹果向量前 5 个数字:", vectors[0][:5])
print("香蕉向量前 5 个数字:", vectors[1][:5])
print("汽车向量前 5 个数字:", vectors[2][:5])


import math

def cosine_similarity(a, b):
    """余弦相似度：1 表示完全一样，0 表示完全无关"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)

apple, banana, car = vectors
print(f"苹果 vs 香蕉 相似度: {cosine_similarity(apple, banana):.4f}")
print(f"苹果 vs 汽车 相似度: {cosine_similarity(apple, car):.4f}")
print(f"香蕉 vs 汽车 相似度: {cosine_similarity(banana, car):.4f}")
