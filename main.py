import numpy as np
import pika, sys, os, json
from clustering import Clustering
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
from vdb import VDB

load_dotenv()


NREC_MAX = 300  # 1000
n_received = 0

clustering_bucket_fv = []
clustering_bucket_id = []

vdb = VDB(os.environ["MILVUS_URI"], os.environ["MILVUS_USERNAME"], os.environ["MILVUS_PASSWORD"], 'image_cats')

client = MongoClient(os.environ["MONGODB_URL"])
db = client.autorec


def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.environ["MQ_HOST"]))
    channel = connection.channel()

    channel.queue_declare(queue='clustering_queue')

    def callback(ch, method, properties, body):
        global n_received
        global clustering_bucket_fv
        global clustering_bucket_id

        body_str = body.decode('utf-8')
        message = json.loads(body_str)
        n_received += 1

        if n_received >= NREC_MAX:
            # clustering
            print('start clustering')
            clustering = Clustering()
            clusters = clustering.clusterize(clustering_bucket_fv)
            
            clusters_idxs = np.argwhere(clusters != -1).flatten()

            if len(clusters_idxs) > 0:  # there are clusters
                cluster_ys = clusters[clusters_idxs]
                cluster_labels = np.unique(cluster_ys)
                print(f'{cluster_labels.size} clusters found')

                for i, cluster_label in enumerate(cluster_labels):
                    cluster_idxs = np.argwhere(clusters == cluster_label).flatten()

                    # find 'centroid' of a cluster
                    f_data_cluster = np.array(clustering_bucket_fv)[cluster_idxs]
                    cluster_centroid = np.mean(f_data_cluster, axis=0)
                    
                    # add centroid to VDB
                    vector_id = vdb.insert_vector([cluster_centroid])

                    # add subcategory to DB
                    rv = db["subcategory"].insert_one(
                        {
                            'vector_id': vector_id,
                            'category_id': None
                        }
                    )
                    subcategory_id = str(rv.inserted_id)

                    # TODO: assign subcategory to all clustering_bucket_id[cluster_idxs]
                    image_idxs = np.array(clustering_bucket_id)[cluster_idxs]
                    for image_id in image_idxs:
                        db["images"].find_one_and_update(
                            {'_id': ObjectId(image_id)},
                            {'$set': {'subcategory': subcategory_id}}
                        )

                
                # save noise as future clustering_bucket
                noise_idxs = np.argwhere(clusters == -1).flatten()
                clustering_bucket_fv = np.array(clustering_bucket_fv)[noise_idxs].tolist()
                clustering_bucket_id = np.array(clustering_bucket_id)[noise_idxs].tolist()
                n_received = 0
            else:
                print('no clusters found')
        else:
            # collect features
            print('collect features')
            clustering_bucket_id.append(message['image_id'])
            clustering_bucket_fv.append(message['feature_vector'])

        print(" [x] Received %r" % message['image_id'])

    channel.basic_consume(queue='clustering_queue', on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
