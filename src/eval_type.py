import tensorflow as tf
import pprint
import json
import numpy as np
import math
from dataset import KnowledgeGraph

pp = pprint.PrettyPrinter(indent=4)

# init_op = tf.initialize_all_variables()
# init_op_t = tf.initialize_all_tables()


# saver = tf.train.Saver()

ent_data = open('/Users/mo/Downloads/Entity.json').read()
gov_ents = json.loads(ent_data)

l = gov_ents.get('records')
l.insert(0, {
  "id": "63fbed08-b4b5-4881-9636-e7c18d1fccda"
})
kg = KnowledgeGraph(data_dir='./data/GOV/')

builder = tf.saved_model.builder.SavedModelBuilder('./GOV-g2/model-98800/')

with tf.Session() as sess:
  print('------started session------')
  
  # tf.saved_model.loader.load(sess, ['serve'], )
  builder.add_meta_graph_and_variables(sess,
                                       ['serve'],
                                      #  signature_def_map=foo_signatures,
                                      #  assets_collection=foo_assets,
                                       strip_default_attrs=True)

  print('------model loaded------')
  eidxs = []
  for e in gov_ents.get('records'):
    eidxs.append(kg.entity_dict.get(e.get('id')))

  # pp.pprint(eidxs)

  # idTensor = tf.convert_to_tensor(eidxs, tf.int32)
  embedding_dim = 100
  # bound = 6 / math.sqrt(embedding_dim)
  with tf.variable_scope('embedding'):

    entity_embedding = tf.get_variable(name='entity',
                                        shape=[kg.n_entity, embedding_dim],
                                        # initializer=None,
                                        trainable=False,
                                        dtype=tf.float32)
    relation_embedding = tf.get_variable(name='relation',
                                        shape=[kg.n_relation, embedding_dim],
                                        # initializer=None,
                                        trainable=False,
                                        dtype=tf.float32)

    print('-----Initializing tf graph-----')
    # tf.global_variables_initializer().run()
    # tf.local_variables_initializer().run()
    print('-----Initialization accomplished-----')


    entity_emb = entity_embedding.eval(session=sess)
    relation_emb = relation_embedding.eval(session=sess)

    entity_types = tf.nn.embedding_lookup(entity_emb, [56585, 56586, 56587, 56588])
    
    rel = tf.nn.embedding_lookup(relation_emb, kg.relation_dict.get('entity_type'))
  
    type_names = ['Legal Entity',
      'Foreign fund',
      'Fund Entity',
      'Sub-Fund Entity']
  
    pp.pprint(type_names)
    for eidx in eidxs:
      froment = tf.nn.embedding_lookup(entity_embedding, eidx)
      pp.pprint(froment.eval(session=sess)[0:4])
      losses = tf.abs(tf.reduce_sum(tf.subtract(tf.add(froment, rel), entity_types), axis=1))
      modelidx = tf.argmin(losses)
      
      # print(eidx, type_names[modelidx.eval(session=sess)], losses.eval(session=sess))