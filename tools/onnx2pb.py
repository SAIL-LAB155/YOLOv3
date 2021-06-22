import onnx
import onnxruntime
from onnx_tf.backend import prepare
import tensorflow as tf
onnx_filename = ''
# step 2, create onnx_model using tensorflow as backend. check if right and export graph.
onnx_model = onnx.load(onnx_filename)
tf_rep = prepare(onnx_model, strict=False)
# install onnx-tensorflow from github，and tf_rep = prepare(onnx_model, strict=False)
# Reference https://github.com/onnx/onnx-tensorflow/issues/167
# tf_rep = prepare(onnx_model) # whthout strict=False leads to KeyError: 'pyfunc_0'
image = Image.open('pants.jpg')
# debug, here using the same input to check onnx and tf.
output_pytorch, img_np = modelhandle.process(image)
print('output_pytorch = {}'.format(output_pytorch))
output_onnx_tf = tf_rep.run(img_np)
print('output_onnx_tf = {}'.format(output_onnx_tf))
# onnx --> tf.graph.pb
tf_pb_path = onnx_filename + '_graph.pb'
tf_rep.export_graph(tf_pb_path)

# step 3, check if tf.pb is right.
with tf.Graph().as_default():
    graph_def = tf.GraphDef()
    with open(tf_pb_path, "rb") as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
    with tf.Session() as sess:
        # init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()
        # sess.run(init)

        # print all ops, check input/output tensor name.
        # uncomment it if you donnot know io tensor names.
        '''
        print('-------------ops---------------------')
        op = sess.graph.get_operations()
        for m in op:
            print(m.values())
        print('-------------ops done.---------------------')
        '''

        input_x = sess.graph.get_tensor_by_name("0:0")  # input
        outputs1 = sess.graph.get_tensor_by_name('add_1:0')  # 5
        outputs2 = sess.graph.get_tensor_by_name('add_3:0')  # 10
        output_tf_pb = sess.run([outputs1, outputs2], feed_dict={input_x: img_np})
        # output_tf_pb = sess.run([outputs1, outputs2], feed_dict={input_x:np.random.randn(1, 3, 224, 224)})
        print('output_tf_pb = {}'.format(output_tf_pb))