from styx_msgs.msg import TrafficLight
import tensorflow as tf
import rospkg
import numpy as np

class TLClassifier(object):
    def __init__(self, is_site):
        self.model = None
        
        r = rospkg.RosPack()
        path = r.get_path('tl_detector')
        print("DEBUG: Path to tl_detector dir:\n")
        print(path)
        if is_site:
            # load Real image model
            print("DEBUG: Loading Traffic Light Detection Model for detecting Real images\n")
            model_path = path + "/models/frozen_inference_graph.pb"
        else:
            # load simulator image model
            print("DEBUG: Loading Traffic Light Detection Model for detecting Sim images\n")
            #model_path = path + "/models/frozen_inference_graph.sim.pb"  
            model_path = path + "/models/frozen_inference_graph.pb"  

        print("DEBUG: Path to tl_detector model:\n")
        print(model_path)

        self.model = self.load_graph(model_path)
        self.session = tf.Session(graph=self.model)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction
        #return TrafficLight.UNKNOWN

        # TODO: what do the models return? do they return the probs / logits? 
        # or do they already select and return the highest probability state??  
        # or is this different for each of our models (i.e. for site and for sim)
 
        print("DEBUG: About to call Traffic Light Classifier")

        with self.model.as_default():
            with self.session as sess:
                # TODO: are these correct for our models? most prob not. getinfo from Damian
                # TODO: I assume it will be different for site and sim. 
                # then better to write two "get_classification" functions
                #Damian comment: These are correct. They will be the same for Real & Sim Models
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detect_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detect_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detect_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                
                #Expand dimensions since the model expects image to have shape : [1, None, None,3]
                image_np_expanded = np.expand_dims(image, axis=0)

                (boxes, scores, classes, num) = sess.run([detect_boxes, detect_scores, detect_classes, num_detections], 
                                   feed_dict={image_tensor: image_np_expanded})

                best_index = np.argmax(scores[0])
                prediction = classes[0][best_index]
                if scores[0][best_index] > 0.5:   #TODO: tune this threshold
                    if (prediction == 1):
                        detected_light = TrafficLight.RED
                    elif (prediction == 2):
                        detected_light = TrafficLight.GREEN
                    elif (prediction == 3):
                        detected_light = TrafficLight.YELLOW
                    else:
                        detected_light = TrafficLight.UNKNOWN
                else:
                    detected_light = TrafficLight.UNKNOWN

        print("DEBUG: Traffic Classifier results:\n")
        print("classes")
        print("scores")

        # TODO: make sure the enumeration used in models are the same as in styx_msgs/TrafficLight
        return detected_light


    # CODE FROM https://github.com/alex-lechner/Traffic-Light-Classification
    def load_graph(self, graph_file):
        #"""Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph
