# -*- coding: utf-8 -*-
"""全链路回归测试：加载 → 切分 → 检索 → 生成"""
print('========== 全链路回归测试 ==========')
print()

print('[1/4] 加载文档')
from app.loader import load_document
docs = load_document('data/samples/员工手册.pdf')
print(f'  员工手册.pdf: {len(docs)} 页/块')

print()
print('[2/4] 切分')
from app.splitter import split_documents
chunks = split_documents(docs)
print(f'  切成 {len(chunks)} 个小块')

print()
print('[3/4] 检索')
from app.retriever import search
for q in ['年假有几天', '如何做红烧肉', '智能温控杯怎么充电']:
    r = search(q, top_k=3)
    print(f'  问"{q}" -> {len(r)} 个片段')

print()
print('[4/4] 生成答案')
from app.chain import ask
answer, docs = ask('年假有几天？', top_k=3)
print(f'  答案: {answer[:60]}...')

print()
print('========== 全链路回归通过 ==========')
