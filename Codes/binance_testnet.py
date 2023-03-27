from binance.cm_futures import CMFutures

cm_futures_client = CMFutures()

# get server time
print(cm_futures_client.time())

# cm_futures_client = CMFutures(
#     key="69fbff75043f5b471fdcdc10b4318f9e1da91c02f570c6cd0d46a2cc9b3c0dd6",
#     secret="1df35b3c2f058378550a17e7705c94de19112cd8a7dbeb829a662eabb0023070",
# )

# # Get account information
# print(cm_futures_client.account())
