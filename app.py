import streamlit as st
from openvino.inference_engine import IECore

st.title("hello!")
ie = IECore()
model_path = "model/model.xml"
ie_net = ie.read_network(model=model_path, weights=model_path.replace("xml", "bin"))
exec_net = ie.load_network(network=ie_net, num_requests=1, device_name="CPU")

st.markdown(exec_net)
