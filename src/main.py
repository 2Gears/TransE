from dataset import KnowledgeGraph
from model import TransE

import tensorflow as tf
import argparse


def main():
  parser = argparse.ArgumentParser(description='TransE')
  parser.add_argument('--data_dir', type=str, default='../data/FB15k/')
  parser.add_argument('--embedding_dim', type=int, default=200)
  parser.add_argument('--margin_value', type=float, default=1.0)
  parser.add_argument('--score_func', type=str, default='L1')
  parser.add_argument('--batch_size', type=int, default=4800)
  parser.add_argument('--learning_rate', type=float, default=0.001)
  parser.add_argument('--n_generator', type=int, default=24)
  parser.add_argument('--n_rank_calculator', type=int, default=24)
  parser.add_argument('--ckpt_dir', type=str, default='../ckpt/')
  parser.add_argument('--model_name', type=str)
  parser.add_argument('--summary_dir', type=str, default='../summary/')
  parser.add_argument('--max_epoch', type=int, default=500)
  parser.add_argument('--eval_freq', type=int, default=10)
  args = parser.parse_args()
  print(args)
  kg = KnowledgeGraph(data_dir=args.data_dir)
  kge_model = TransE(kg=kg, embedding_dim=args.embedding_dim, margin_value=args.margin_value,
                    score_func=args.score_func, batch_size=args.batch_size, learning_rate=args.learning_rate,
                    n_generator=args.n_generator, n_rank_calculator=args.n_rank_calculator,
                    model_name=args.model_name, ckpt_dir=args.ckpt_dir)
  gpu_config = tf.GPUOptions(allow_growth=False)
  sess_config = tf.ConfigProto(gpu_options=gpu_config)
  with tf.Session(config=sess_config) as sess:
    print('-----Initializing tf graph-----')
    tf.global_variables_initializer().run()
    print('-----Initialization accomplished-----')
    kge_model.check_norm(session=sess)
    summary_writer = tf.summary.FileWriter(
      logdir=args.summary_dir, graph=sess.graph)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)
    for epoch in range(args.max_epoch):
      print('=' * 30 + '[EPOCH {}]'.format(epoch) + '=' * 30)
      kge_model.launch_training(
        session=sess, summary_writer=summary_writer)
      if (epoch + 1) % args.eval_freq == 0:
        kge_model.launch_evaluation(session=sess, saver=saver)
      
      print('-----Save checkpoint-----')
      step_str = str(kge_model.global_step.eval(session=sess))
      save_path = args.ckpt_dir + '/' + args.model_name + step_str + '.ckpt'
      saver_path = saver.save(sess, save_path)
      tf.saved_model.simple_save(sess, args.ckpt_dir + '/model-' + step_str, inputs={'triple': kge_model.eval_triple}, outputs={
                                 'entity-embedding': kge_model.entity_embedding, 'relation-embedding': kge_model.relation_embedding})

      print("Model saved in path: %s" % saver_path)



if __name__ == '__main__':
  main()
