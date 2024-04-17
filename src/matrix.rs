use rand::{thread_rng,Rng};
use std::fmt::{Debug, Formatter, Result};
// use std::fmt::{Debug, Formatter, Result};

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}
impl Matrix {
    pub fn zeros(rows:usize,cols:usize) -> Matrix{
        Matrix{
            rows,
            cols,
            data: vec![vec![0.0;cols];rows], 
        }
    }

    pub fn random(rows:usize,cols:usize) -> Matrix{
        let mut rng = thread_rng();
        let mut res = Matrix::zeros(rows,cols);
        for i in 0..rows{
            for j in 0..cols{
                res.data[i][j] = rng.gen::<f64>() * 2.0 - 1.0;
            }
        }
        res
    }

    pub fn from(data: Vec<Vec<f64>>) -> Matrix {
		Matrix {
			rows: data.len(),
			cols: data[0].len(),
			data,
		}
	}

    pub fn multiply(&mut self, other: &Matrix) -> Matrix{
        if self.cols != other.rows {
            panic!("Different matrix orders");
        }
        let mut res = Matrix::zeros(self.rows,other.cols);
        for i in 0..self.rows{
            for j in 0..other.cols{
                for k in 0..self.cols{
                    res.data[i][j] += self.data[i][k] * other.data[k][j] ;
                }
            }
        }
        res
    }

    pub fn add(&mut self, other: &Matrix) -> Matrix{
        if (self.cols != other.cols) || (self.rows!=other.rows) {
            panic!("Different matrix orders");
        }
        let mut res = Matrix::zeros(self.rows,other.cols);
        for i in 0..self.rows{
            for j in 0..other.cols{
                    res.data[i][j] += self.data[i][j] + other.data[i][j] ;
            }
        }
        res
    }

    pub fn dot_multiply(&mut self, other: &Matrix) -> Matrix{
        if (self.cols != other.cols) || (self.rows!=other.rows) {
            panic!("Different matrix orders");
        }
        let mut res = Matrix::zeros(self.rows,other.cols);
        for i in 0..self.rows{
            for j in 0..other.cols{
                    res.data[i][j] += self.data[i][j] * other.data[i][j] ;
            }
        }
        res
    }

    pub fn subtract(&mut self, other: &Matrix) -> Matrix{
        if (self.cols != other.cols) || (self.rows!=other.rows) {
            panic!("Different matrix orders");
        }
        let mut res = Matrix::zeros(self.rows,other.cols);
        for i in 0..self.rows{
            for j in 0..other.cols{
                    res.data[i][j] += self.data[i][j] - other.data[i][j] ;
            }
        }
        res
    }

    pub fn map(&self, function: &dyn Fn(f64) -> f64) -> Matrix {
		Matrix::from(
			(self.data)
				.clone()
				.into_iter()
				.map(|row| row.into_iter().map(|value| function(value)).collect())
				.collect(),
		)
	}

    pub fn transpose(&mut self) -> Matrix{
        let mut res = Matrix::zeros(self.cols,self.rows);
        for i in 0..self.rows{
            for j in 0..self.cols{
                    res.data[j][i] = self.data[i][j] ;
            }
        }
        res
    }

}

impl Debug for Matrix {
	fn fmt(&self, f: &mut Formatter) -> Result {
		write!(
			f,
			"Matrix {{\n{}\n}}",
			(&self.data)
				.into_iter()
				.map(|row| "  ".to_string()
					+ &row
						.into_iter()
						.map(|value| value.to_string())
						.collect::<Vec<String>>()
						.join(" "))
				.collect::<Vec<String>>()
				.join("\n")
		)
	}
}