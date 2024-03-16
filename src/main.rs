use rand;
use rand::distributions::{Uniform, Distribution};


fn soft_max(z:Vec<f64>) -> Vec<f64>{
    let sum:f64 = z.iter().map(|x| x.exp()).sum();
    z.iter().map(|x| x.exp()/sum).collect()
}

struct Layer{
    pub nodes:Vec<f64>,
    pub activ:Vec<f64>,
    pub biases:Vec<f64>,
    pub weights:Vec<Vec<f64>>,
}

struct Network{
    pub layers:Vec<Layer>
}
impl Network{
    pub fn new(layer_sizes:Vec<usize>) -> Self{
        let mut network:Self = Self { layers: vec![] };
        let mut prev_layer_size = 0usize;
        let range = Uniform::from(-1.0 .. 1.0);
        let mut rng = rand::thread_rng();
        for i in layer_sizes{

            let l:Layer = Layer { 
                nodes: vec![0.0;i], 
                activ: vec![0.0;i],
                biases: (0..i).map(|_| if prev_layer_size > 0 {range.sample(&mut rng)} else {0.0}).collect(), 
                weights: (0..i).map(|_| (0..prev_layer_size).map(|_| range.sample(&mut rng)).collect()).collect() };
            network.layers.push(l);
            prev_layer_size = i;
        }
        network
    }

    pub fn num_layers(&self) -> usize{
        self.layers.len()
    }
}
fn sigmoid(v: f64) -> f64 {
    if v < -40.0 {
        0.0
    } else if v > 40.0 {
        1.0
    } else {
        1.0 / (1.0 + f64::exp(-v))
    }
}
fn prop(network:&mut Network){
    assert!(network.num_layers() > 1, "Invalid network size!");
    //start at first hidden layer, "pull" in from prior layer
    for layer_idx in 1..network.num_layers(){
        for node_idx in 0..network.layers[layer_idx].nodes.len(){
            println!("On node: {node_idx} of layer: {layer_idx}");
            let weights = &network.layers[layer_idx].weights[node_idx];
            let bias = &network.layers[layer_idx].biases[node_idx];
            let mut prev_node_idx = 0;
            //iter through all nodes of prev layer
            let sum:f64 = network.layers[layer_idx-1].nodes.iter().map(|val| {
                //multiply weight by val, add bias
                let cur_sum = weights[prev_node_idx] * val + bias;
                prev_node_idx += 1;
                cur_sum
            }).sum(); // <= this is the values of all the nodes in the previous layer multiplied by their weights with bias added
            println!("Sum: {sum}");
            network.layers[layer_idx].nodes[node_idx] = sum;
            network.layers[layer_idx].activ[node_idx] = sigmoid(sum); //<= activation function
        }
    }
}

fn main() {
    let mut n = Network::new(vec![74usize,100usize,60usize,10usize]);
    for i in 0..n.layers[0].nodes.len(){
        n.layers[0].nodes[i] = sigmoid(rand::random());
    }
    let mut layer_num = 0;
    for l in &n.layers{
        layer_num += 1;
        println!("======== LAYER {layer_num} ========");
        println!("nodes: ");
        l.nodes.iter().for_each(|i| print!("{i}, "));
        println!();

        println!("activ: ");
        l.activ.iter().for_each(|i| print!("{i}, "));
        println!();

        println!("biases: ");
        l.biases.iter().for_each(|i| print!("{i}, "));
        println!();

        println!("weights: ");
        l.weights.iter().for_each(|i| print!("{}, ", i.len()));
        println!();
    }
    prop(&mut n);
    layer_num = 0;
    for l in &n.layers{
        layer_num += 1;
        println!("======== LAYER {layer_num} ========");
        println!("nodes: ");
        l.nodes.iter().for_each(|i| print!("{i}, "));
        println!();

        println!("activ: ");
        l.activ.iter().for_each(|i| print!("{i}, "));
        println!();

        println!("biases: ");
        l.biases.iter().for_each(|i| print!("{i}, "));
        println!();

        println!("weights: ");
        l.weights.iter().for_each(|i| print!("{}, ", i.len()));
        println!();
    }
}
