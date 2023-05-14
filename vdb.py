from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema


class VDB:
    def __init__(self, milvus_uri, user, password, collection_name):
        connections.connect(
            "default",
            uri=milvus_uri,
            user=user,
            password=password,
            secure=True
        )

        self.collection_name = collection_name

        id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True)
        fv_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1280)

        self.schema = CollectionSchema(
            fields=[id_field, fv_field],
            auto_id=True,
            description="image classification data"
        )

        self.collection = Collection(name=self.collection_name, schema=self.schema)

        self.collection.load()

        self.search_params = {"metric_type": "L2"}

    
    # def get_knn(self, q_vec: list[float], topk: int = 1):
    #     results = self.collection.search(
    #         [q_vec],
    #         anns_field='vector',
    #         param=self.search_params,
    #         limit=topk,
    #         guarantee_timestamp=1
    #     )
    #     return results

    def insert_vector(self, q_vec: list[float]):
        mr = self.collection.insert([q_vec])
        vector_id = mr.primary_keys[0]
        return vector_id


if __name__ == '__main__':
    import os
    import numpy as np
    from dotenv import load_dotenv

    load_dotenv()

    vdb = VDB(os.environ["MILVUS_URI"], os.environ["MILVUS_USERNAME"], os.environ["MILVUS_PASSWORD"], 'image_cats')

    vdb.insert_vector([np.random.randn(1280).astype(np.float32)])




