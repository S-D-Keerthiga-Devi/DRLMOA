import React, { useState } from 'react';
import { RefreshCw, Play } from 'lucide-react';
import { Card } from './ui/Card';
import Slider from './ui/Slider';
import CityGraph from './visualization/CityGraph';

const ProblemGenerator: React.FC = () => {
  const [problemType, setProblemType] = useState<'euclidean' | 'mixed'>('euclidean');
  const [numCities, setNumCities] = useState<number>(40);
  const [numInstances, setNumInstances] = useState<number>(1000);
  const [distribution, setDistribution] = useState<'uniform' | 'normal' | 'cluster'>('uniform');
  const [generating, setGenerating] = useState(false);
  const [generated, setGenerated] = useState(false);
  
  const handleGenerate = () => {
    setGenerating(true);
    // Simulate generation process
    setTimeout(() => {
      setGenerating(false);
      setGenerated(true);
    }, 1500);
  };

  const generateNewInstance = () => {
    // Would actually regenerate a new instance
    // For now just toggle to simulate
    setGenerated(false);
    setTimeout(() => setGenerated(true), 300);
  };

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">Problem Generator</h2>
      <p className="mb-6">
        Generate problem instances for training the DRL-MOA model. The paper used 500,000 instances of 40 cities for Euclidean bi-objective TSP and 120,000 instances for Mixed bi-objective TSP.
      </p>

      <div className="grid md:grid-cols-3 gap-6">
        <div className="md:col-span-1 space-y-6">
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Problem Configuration</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Problem Type</label>
                <div className="flex rounded-md overflow-hidden">
                  <button
                    className={`flex-1 py-2 ${problemType === 'euclidean' 
                      ? 'bg-blue-500 text-white' 
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200'}`}
                    onClick={() => setProblemType('euclidean')}
                  >
                    Euclidean
                  </button>
                  <button
                    className={`flex-1 py-2 ${problemType === 'mixed' 
                      ? 'bg-blue-500 text-white' 
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200'}`}
                    onClick={() => setProblemType('mixed')}
                  >
                    Mixed
                  </button>
                </div>
              </div>
              
              <Slider 
                label="Number of Cities"
                value={numCities}
                onChange={setNumCities}
                min={10}
                max={200}
                step={10}
              />
              
              <Slider 
                label="Number of Instances"
                value={numInstances}
                onChange={setNumInstances}
                min={100}
                max={500000}
                step={100}
                suffix="instances"
                format={(val) => val >= 1000 ? `${val/1000}k` : val.toString()}
              />
              
              <div>
                <label className="block text-sm font-medium mb-2">Distribution</label>
                <select 
                  className="w-full p-2 border rounded dark:bg-gray-700 dark:border-gray-600"
                  value={distribution}
                  onChange={(e) => setDistribution(e.target.value as any)}
                >
                  <option value="uniform">Uniform</option>
                  <option value="normal">Normal</option>
                  <option value="cluster">Clustered</option>
                </select>
              </div>
            </div>
            
            <div className="mt-6">
              <button
                className="w-full py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-md flex items-center justify-center transition-colors disabled:opacity-50 disabled:pointer-events-none"
                onClick={handleGenerate}
                disabled={generating}
              >
                {generating ? (
                  <>
                    <div className="spinner mr-2"></div>
                    Generating...
                  </>
                ) : (
                  <>
                    <Play size={18} className="mr-2" />
                    Generate Instances
                  </>
                )}
              </button>
            </div>
          </Card>
          
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Problem Summary</h3>
            {generated ? (
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Problem Type:</span>
                  <span className="font-medium">{problemType === 'euclidean' ? 'Euclidean Bi-Objective' : 'Mixed Bi-Objective'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Input Dimension:</span>
                  <span className="font-medium">{problemType === 'euclidean' ? '4D (x1,y1,x2,y2)' : '3D (x,y,z)'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Instances:</span>
                  <span className="font-medium">{numInstances >= 1000 ? `${numInstances/1000}k` : numInstances}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Distribution:</span>
                  <span className="font-medium capitalize">{distribution}</span>
                </div>
                <div className="pt-2">
                  <button 
                    className="text-blue-500 hover:text-blue-600 flex items-center text-sm"
                    onClick={generateNewInstance}
                  >
                    <RefreshCw size={14} className="mr-1" />
                    Generate new example
                  </button>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                <p>Generate instances to see problem summary</p>
              </div>
            )}
          </Card>
        </div>
        
        <div className="md:col-span-2">
          <Card className="p-6 h-full">
            <div className="flex justify-between mb-4">
              <h3 className="text-lg font-semibold">Problem Visualization</h3>
              {generated && (
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    Objective 1
                  </span>
                  {problemType === 'euclidean' && (
                    <>
                      <div className="h-5 border-l border-gray-300 dark:border-gray-600"></div>
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        Objective 2
                      </span>
                    </>
                  )}
                </div>
              )}
            </div>
            
            {generated ? (
              <div className="grid md:grid-cols-2 gap-4 h-[500px]">
                <div className="h-full">
                  <CityGraph title="Objective 1 (Euclidean Distance)" numCities={numCities} />
                </div>
                
                {problemType === 'euclidean' ? (
                  <div className="h-full">
                    <CityGraph title="Objective 2 (Euclidean Distance)" numCities={numCities} />
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center">
                    <table className="w-full">
                      <thead>
                        <tr className="bg-gray-50 dark:bg-gray-800">
                          <th className="py-2 text-left">City</th>
                          <th className="py-2 text-left">Value (Obj 2)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[...Array(10)].map((_, i) => (
                          <tr key={i} className="border-t dark:border-gray-700">
                            <td className="py-2">City {i + 1}</td>
                            <td className="py-2">{(Math.random() * 100).toFixed(2)}</td>
                          </tr>
                        ))}
                        <tr>
                          <td colSpan={2} className="py-2 text-center text-gray-500 dark:text-gray-400">
                            ... and {numCities - 10} more cities
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-center h-[500px] text-gray-500 dark:text-gray-400">
                <p>Generate instances to visualize problem</p>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
};

export default ProblemGenerator;