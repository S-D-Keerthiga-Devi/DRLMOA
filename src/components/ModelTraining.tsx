import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Save } from 'lucide-react';
import { Card } from './ui/Card';
import Slider from './ui/Slider';
import { LineChart } from './visualization/LineChart';

const ModelTraining: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [epoch, setEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(5);
  const [learningRate, setLearningRate] = useState(0.001);
  const [batchSize, setBatchSize] = useState(64);
  const [hiddenDim, setHiddenDim] = useState(128);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [accuracyHistory, setAccuracyHistory] = useState<number[]>([]);
  
  // Simulate training progress
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isTraining) {
      interval = setInterval(() => {
        setProgress(prev => {
          const newProgress = prev + 0.5;
          
          // Add loss and accuracy data points
          if (Math.floor(newProgress) > Math.floor(prev)) {
            setLossHistory(prevLoss => {
              const lastValue = prevLoss.length > 0 ? prevLoss[prevLoss.length - 1] : 3;
              return [...prevLoss, Math.max(0.1, lastValue * 0.95 - Math.random() * 0.1)];
            });
            
            setAccuracyHistory(prevAcc => {
              const lastValue = prevAcc.length > 0 ? prevAcc[prevAcc.length - 1] : 0.5;
              return [...prevAcc, Math.min(0.99, lastValue * 1.02 + Math.random() * 0.03)];
            });
          }
          
          // Update epoch
          const currentEpoch = Math.floor((newProgress / 100) * totalEpochs);
          if (currentEpoch !== epoch) {
            setEpoch(currentEpoch);
          }
          
          // Reset if training completes
          if (newProgress >= 100) {
            setIsTraining(false);
            return 100;
          }
          
          return newProgress;
        });
      }, 200);
    }
    
    return () => clearInterval(interval);
  }, [isTraining, epoch, totalEpochs]);
  
  const handleStartTraining = () => {
    setIsTraining(true);
  };
  
  const handlePauseTraining = () => {
    setIsTraining(false);
  };
  
  const handleResetTraining = () => {
    setIsTraining(false);
    setProgress(0);
    setEpoch(0);
    setLossHistory([]);
    setAccuracyHistory([]);
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">Model Training</h2>
      <p className="mb-6">
        Train the DRL-MOA model using generated problem instances. The paper trained on 500,000 instances of 40 cities for 5 epochs for Euclidean bi-objective TSP.
      </p>

      <div className="grid md:grid-cols-3 gap-6">
        <div className="md:col-span-1 space-y-6">
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Training Parameters</h3>
            
            <div className="space-y-4">
              <Slider 
                label="Learning Rate"
                value={learningRate}
                onChange={setLearningRate}
                min={0.0001}
                max={0.01}
                step={0.0001}
                format={(val) => val.toExponential(4)}
              />
              
              <Slider 
                label="Batch Size"
                value={batchSize}
                onChange={setBatchSize}
                min={8}
                max={256}
                step={8}
              />
              
              <Slider 
                label="Hidden Dimension"
                value={hiddenDim}
                onChange={setHiddenDim}
                min={32}
                max={512}
                step={16}
              />
              
              <Slider 
                label="Total Epochs"
                value={totalEpochs}
                onChange={setTotalEpochs}
                min={1}
                max={20}
                step={1}
              />
              
              <div>
                <label className="block text-sm font-medium mb-2">Model Architecture</label>
                <select className="w-full p-2 border rounded dark:bg-gray-700 dark:border-gray-600">
                  <option>Attention-based Encoder + LSTM</option>
                  <option>Transformer Encoder + LSTM</option>
                  <option>GNN + LSTM</option>
                </select>
              </div>
            </div>
            
            <div className="mt-6 space-y-2">
              {!isTraining ? (
                <button
                  className="w-full py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-md flex items-center justify-center"
                  onClick={handleStartTraining}
                  disabled={progress === 100}
                >
                  <Play size={18} className="mr-2" />
                  {progress === 0 ? 'Start Training' : 'Resume Training'}
                </button>
              ) : (
                <button
                  className="w-full py-2 bg-amber-500 hover:bg-amber-600 text-white rounded-md flex items-center justify-center"
                  onClick={handlePauseTraining}
                >
                  <Pause size={18} className="mr-2" />
                  Pause
                </button>
              )}
              
              <button
                className="w-full py-2 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-800 dark:text-gray-200 rounded-md flex items-center justify-center"
                onClick={handleResetTraining}
                disabled={progress === 0}
              >
                <RotateCcw size={18} className="mr-2" />
                Reset
              </button>
              
              {progress === 100 && (
                <button
                  className="w-full py-2 bg-green-500 hover:bg-green-600 text-white rounded-md flex items-center justify-center"
                >
                  <Save size={18} className="mr-2" />
                  Save Model
                </button>
              )}
            </div>
          </Card>
          
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Training Progress</h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Progress</span>
                  <span>{progress.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                  <div 
                    className="bg-blue-500 h-2.5 rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Epoch:</span>
                <span className="font-medium">{epoch} / {totalEpochs}</span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Learning Rate:</span>
                <span className="font-medium">{learningRate.toExponential(4)}</span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Batch Size:</span>
                <span className="font-medium">{batchSize}</span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Current Loss:</span>
                <span className="font-medium">
                  {lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].toFixed(4) : '-'}
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Current Accuracy:</span>
                <span className="font-medium">
                  {accuracyHistory.length > 0 ? (accuracyHistory[accuracyHistory.length - 1] * 100).toFixed(2) + '%' : '-'}
                </span>
              </div>
            </div>
          </Card>
        </div>
        
        <div className="md:col-span-2 space-y-6">
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Training Metrics</h3>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div className="h-64">
                <h4 className="text-sm font-medium mb-2">Loss</h4>
                <LineChart 
                  data={lossHistory}
                  labels={lossHistory.map((_, i) => i + 1)}
                  color="#3B82F6"
                  yAxisMin={0}
                />
              </div>
              
              <div className="h-64">
                <h4 className="text-sm font-medium mb-2">Accuracy</h4>
                <LineChart 
                  data={accuracyHistory}
                  labels={accuracyHistory.map((_, i) => i + 1)}
                  color="#10B981"
                  yAxisMin={0}
                  yAxisMax={1}
                  formatYAxis={(value) => `${(value * 100).toFixed(0)}%`}
                />
              </div>
            </div>
          </Card>
          
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Model Architecture</h3>
            <div className="overflow-auto">
              <div className="flex justify-center py-4">
                <svg width="600" height="240" className="dark:invert-[0.85]">
                  {/* Input Layer */}
                  <g transform="translate(50, 30)">
                    <rect width="100" height="180" rx="5" fill="#EFF6FF" stroke="#3B82F6" strokeWidth="1.5"/>
                    <text x="50" y="-10" textAnchor="middle" fontSize="14" fontWeight="500">Input</text>
                    {[...Array(5)].map((_, i) => (
                      <React.Fragment key={`input-${i}`}>
                        <circle cx="50" cy={30 + i * 30} r="12" fill="#3B82F6" />
                        <text x="50" cy={30 + i * 30} textAnchor="middle" fontSize="10" fill="white" dy="4">
                          {i < 4 ? `x${i+1}` : '...'}
                        </text>
                      </React.Fragment>
                    ))}
                  </g>
                  
                  {/* Encoder */}
                  <g transform="translate(200, 30)">
                    <rect width="100" height="180" rx="5" fill="#EFF6FF" stroke="#3B82F6" strokeWidth="1.5"/>
                    <text x="50" y="-10" textAnchor="middle" fontSize="14" fontWeight="500">Encoder</text>
                    {[...Array(5)].map((_, i) => (
                      <React.Fragment key={`encoder-${i}`}>
                        <circle cx="50" cy={30 + i * 30} r="12" fill="#8B5CF6" />
                        <text x="50" cy={30 + i * 30} textAnchor="middle" fontSize="10" fill="white" dy="4">
                          {i < 4 ? `h${i+1}` : '...'}
                        </text>
                      </React.Fragment>
                    ))}
                  </g>
                  
                  {/* LSTM Decoder */}
                  <g transform="translate(350, 30)">
                    <rect width="100" height="180" rx="5" fill="#EFF6FF" stroke="#3B82F6" strokeWidth="1.5"/>
                    <text x="50" y="-10" textAnchor="middle" fontSize="14" fontWeight="500">LSTM Decoder</text>
                    {[...Array(5)].map((_, i) => (
                      <React.Fragment key={`decoder-${i}`}>
                        <circle cx="50" cy={30 + i * 30} r="12" fill="#F59E0B" />
                        <text x="50" cy={30 + i * 30} textAnchor="middle" fontSize="10" fill="white" dy="4">
                          {i < 4 ? `d${i+1}` : '...'}
                        </text>
                      </React.Fragment>
                    ))}
                  </g>
                  
                  {/* Output */}
                  <g transform="translate(500, 30)">
                    <rect width="100" height="180" rx="5" fill="#EFF6FF" stroke="#3B82F6" strokeWidth="1.5"/>
                    <text x="50" y="-10" textAnchor="middle" fontSize="14" fontWeight="500">Output</text>
                    {[...Array(5)].map((_, i) => (
                      <React.Fragment key={`output-${i}`}>
                        <circle cx="50" cy={30 + i * 30} r="12" fill="#10B981" />
                        <text x="50" cy={30 + i * 30} textAnchor="middle" fontSize="10" fill="white" dy="4">
                          {i < 4 ? `o${i+1}` : '...'}
                        </text>
                      </React.Fragment>
                    ))}
                  </g>
                  
                  {/* Connections */}
                  <g>
                    {/* Input to Encoder */}
                    <path d="M150,120 C175,120 175,120 200,120" stroke="#3B82F6" strokeWidth="1.5" fill="none" />
                    
                    {/* Encoder to Decoder */}
                    <path d="M300,120 C325,120 325,120 350,120" stroke="#8B5CF6" strokeWidth="1.5" fill="none" />
                    
                    {/* Decoder to Output */}
                    <path d="M450,120 C475,120 475,120 500,120" stroke="#F59E0B" strokeWidth="1.5" fill="none" />
                  </g>
                </svg>
              </div>
            </div>
            
            <div className="mt-4 code-block text-sm">
              <pre>
{`# DRL-MOA Model Architecture (PyTorch-like pseudocode)
class DRLMOA(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads=8):
        super().__init__()
        self.encoder = TransformerEncoder(input_dim, hidden_dim, n_heads)
        self.decoder = LSTMDecoder(hidden_dim)
        self.multi_head_attention = MultiHeadAttention(hidden_dim, n_heads)
        
    def forward(self, state, weights):
        # Encode state
        encoded = self.encoder(state)  # [batch_size, n_cities, hidden_dim]
        
        # Initialize solution
        solution = []
        
        # Decode step by step
        for step in range(n_cities):
            # Calculate attention scores
            scores = self.multi_head_attention(encoded, solution)
            
            # Get probabilities for next city
            probs = self.decoder(scores, encoded)
            
            # Choose next city based on weights
            next_city = self.weighted_choice(probs, weights)
            solution.append(next_city)
            
        return solution`}
              </pre>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default ModelTraining;