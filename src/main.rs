use activations::SIGMOID;
use network::Network;
use std::fs::File;
use std::io::Read;
use zip::ZipArchive;

pub mod activations;
pub mod matrix;
pub mod network;


fn read_npz_file(filename: &str, array_name: &str, size: usize) -> Option<Vec<Vec<f64>>> {
    // Open the NPZ file
    let file = File::open(filename).expect("Failed to open file");
    let mut archive = ZipArchive::new(file).expect("Failed to open zip archive");
    // println!("{:?}",archive);

    // Find the array with the given name
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).expect("Failed to read file in zip archive");
        if file.name() == array_name {
            // Read the array data
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer).expect("Failed to read array data");
            
            // Convert the data into Vec<Vec<f64>>
            let mut data = Vec::new();
            let mut row = Vec::new();
            let mut counter = 0;
            for value in buffer.chunks_exact(8) {
                if counter<16 {counter+=1;}
                else{
                let value = f64::from_le_bytes([
                    value[0], value[1], value[2], value[3],
                    value[4], value[5], value[6], value[7],
                    ]);
                    row.push(value);
                    if row.len() == size   {
                        data.push(row);
                        row = Vec::new();
                    }
                }
            }
                // println!("{:?}",data.len());
                
            return Some(data);
        }
    }

    None
}

// fn main() {
//     let inputs = vec![
//         vec![0.0,0.0],
//         vec![1.0,0.0],
//         vec![0.0,1.0],
//         vec![1.0,1.0],
//     ];

//     let targets = vec![
//         vec![0.0],
//         vec![1.0],
//         vec![1.0],
//         vec![0.0],
//     ];

//     let mut network = Network::new(vec![2,3,1], 0.5, SIGMOID);
//     network.train(inputs,targets,100000);

//     println!("0 and 0: {:?}",network.feed_forward(vec![0.0,0.0]));
//     println!("1 and 0: {:?}",network.feed_forward(vec![1.0,0.0]));
//     println!("0 and 1: {:?}",network.feed_forward(vec![0.0,1.0]));
//     println!("1 and 1: {:?}",network.feed_forward(vec![1.0,1.0]));
// }

fn main() {

    let filename = "src/mnist.npz";
    let training_images = "training_images.npy";
    let training_labels = "training_labels.npy";

    let images_data = read_npz_file(filename, training_images,784).unwrap();
    let labels_data = read_npz_file(filename, training_labels,10).unwrap();
    let inputs = images_data;
    let targets = labels_data;
    let test = inputs[0].clone();

    println!("{:?}",inputs.len());
    println!("{:?}",targets.len());
    let mut network = Network::new(vec![784, 5,10], 0.5, SIGMOID);

    network.train(inputs,targets,1);
    // println!("{:?}",network.feed_forward(test));
}
