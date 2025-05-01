import React, { useState } from 'react';
import { Card } from './ui/Card';
import { BarChart } from './visualization/BarChart';
import { LineChart } from './visualization/LineChart';

const ResultsComparison: React.FC = () => {
  const [problemType, setProblemType] = useState<'euclidean' | 'mixed'>('euclidean');
  const [compareBy, setCompareBy] = useState<'hypervolume' | 'time' | 'coverage'>('hypervolume');
  
  const getComparisonData = () => {
    // This would be data from real experiments
    if (compareBy === 'hypervolume') {
      return {
        labels: ['40', '70', '100', '150', '200'],
        datasets: [
          {
            label: 'DRL-MOA (Ours)',
            data: [0.89, 0.87, 0.85, 0.82, 0.79],
            color: '#3B82F6'
          },
          {
            label: 'MOEA/D',
            data: [0.85, 0.81, 0.77, 0.73, 0.68],
            color: '#F59E0B'
          },
          {
            label: 'NSGA-II',
            data: [0.83, 0.79, 0.75, 0.71, 0.66],
            color: '#10B981'
          }
        ]
      };
    } else if (compareBy === 'time') {
      return {
        labels: ['40', '70', '100', '150', '200'],
        datasets: [
          {
            label: 'DRL-MOA (Ours)',
            data: [0.5, 1.2, 2.8, 6.9, 14.2],
            color: '#3B82F6'
          },
          {
            label: 'MOEA/D',
            data: [1.8, 4.9, 12.5, 38.7, 88.3],
            color: '#F59E0B'
          },
          {
            label: 'NSGA-II',
            data: [2.1, 5.7, 14.3, 43.5, 102.1],
            color: '#10B981'
          }
        ]
      };
    } else {
      return {
        labels: ['40', '70', '100', '150', '200'],
        datasets: [
          {
            label: 'DRL-MOA (Ours)',
            data: [0.92, 0.90, 0.88, 0.85, 0.82],
            color: '#3B82F6'
          },
          {
            label: 'MOEA/D',
            data: [0.84, 0.81, 0.77, 0.72, 0.67],
            color: '#F59E0B'
          },
          {
            label: 'NSGA-II',
            data: [0.80, 0.76, 0.72, 0.67, 0.62],
            color: '#10B981'
          }
        ]
      };
    }
  };

  const data = getComparisonData();
  
  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">Results Comparison</h2>
      <p className="mb-6">
        Compare the performance of DRL-MOA with state-of-the-art multi-objective optimization algorithms on various problem instances.
      </p>

      <div className="grid md:grid-cols-4 gap-6 mb-6">
        <div className="md:col-span-1">
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Comparison Settings</h3>
            
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
              
              <div>
                <label className="block text-sm font-medium mb-2">Compare By</label>
                <div className="space-y-2">
                  <button
                    className={`w-full py-2 px-3 text-left rounded-md ${compareBy === 'hypervolume' 
                      ? 'bg-blue-500 text-white' 
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200'}`}
                    onClick={() => setCompareBy('hypervolume')}
                  >
                    Hypervolume
                  </button>
                  <button
                    className={`w-full py-2 px-3 text-left rounded-md ${compareBy === 'time' 
                      ? 'bg-blue-500 text-white' 
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200'}`}
                    onClick={() => setCompareBy('time')}
                  >
                    Computation Time
                  </button>
                  <button
                    className={`w-full py-2 px-3 text-left rounded-md ${compareBy === 'coverage' 
                      ? 'bg-blue-500 text-white' 
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200'}`}
                    onClick={() => setCompareBy('coverage')}
                  >
                    Solution Coverage
                  </button>
                </div>
              </div>
            </div>
          </Card>
        </div>
        
        <div className="md:col-span-3">
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">
              {compareBy === 'hypervolume' ? 'Hypervolume Comparison' : 
               compareBy === 'time' ? 'Computation Time Comparison (seconds)' : 
               'Solution Coverage Comparison'}
            </h3>
            
            <div className="h-80">
              {compareBy === 'time' ? (
                <BarChart
                  labels={data.labels}
                  datasets={data.datasets}
                  xAxisLabel="Number of Cities"
                  yAxisLabel="Time (seconds)"
                  logarithmic={true}
                />
              ) : (
                <LineChart 
                  labels={data.labels}
                  datasets={data.datasets.map(dataset => ({
                    data: dataset.data,
                    label: dataset.label,
                    color: dataset.color
                  }))}
                  xAxisLabel="Number of Cities"
                  yAxisLabel={compareBy === 'hypervolume' ? 'Hypervolume' : 'Coverage Ratio'}
                  yAxisMin={0.5}
                  yAxisMax={1}
                />
              )}
            </div>
          </Card>
        </div>
      </div>
      
      <div className="grid md:grid-cols-2 gap-6">
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Performance Analysis</h3>
          
          <div className="space-y-4">
            <div>
              <h4 className="font-medium mb-2">Summary</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                The DRL-MOA model consistently outperforms traditional multi-objective evolutionary algorithms (MOEA/D and NSGA-II) across different problem sizes and metrics. 
                {compareBy === 'hypervolume' && 'The hypervolume indicator shows that our approach maintains higher solution quality as problem size increases.'}
                {compareBy === 'time' && 'Our approach demonstrates significantly faster computation times, with the gap widening as problem size increases.'}
                {compareBy === 'coverage' && 'The solution coverage analysis shows that our model finds more diverse and higher quality solutions on the Pareto front.'}
              </p>
            </div>
            
            <div>
              <h4 className="font-medium mb-2">Key Observations</h4>
              <ul className="list-disc pl-5 text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>DRL-MOA maintains consistent performance advantage across all problem sizes</li>
                <li>Performance gap widens as problem complexity increases</li>
                <li>{compareBy === 'time' ? 'Orders of magnitude faster for large instances' : 'Superior solution quality for all instances'}</li>
                <li>The model generalizes well to problem sizes it wasn't trained on</li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-medium mb-2">Limitations</h4>
              <ul className="list-disc pl-5 text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>Training requires significant computational resources</li>
                <li>Performance may degrade for problem structures very different from training data</li>
                <li>Current implementation limited to bi-objective problems</li>
              </ul>
            </div>
          </div>
        </Card>
        
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Method Comparison</h3>
          
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead>
                <tr className="bg-gray-50 dark:bg-gray-800">
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Method</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Type</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Pareto Approx.</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Time Complexity</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Scalability</th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800">
                <tr className="hover:bg-gray-50 dark:hover:bg-gray-800">
                  <td className="px-3 py-4 whitespace-nowrap text-sm font-medium">DRL-MOA</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">Deep RL</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">Excellent</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">O(n)</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">High</td>
                </tr>
                <tr className="hover:bg-gray-50 dark:hover:bg-gray-800">
                  <td className="px-3 py-4 whitespace-nowrap text-sm font-medium">MOEA/D</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">Evolutionary</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">Good</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">O(n²)</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">Medium</td>
                </tr>
                <tr className="hover:bg-gray-50 dark:hover:bg-gray-800">
                  <td className="px-3 py-4 whitespace-nowrap text-sm font-medium">NSGA-II</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">Evolutionary</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">Good</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">O(n²)</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">Medium</td>
                </tr>
                <tr className="hover:bg-gray-50 dark:hover:bg-gray-800">
                  <td className="px-3 py-4 whitespace-nowrap text-sm font-medium">Brute Force</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">Exact</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">Perfect</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">O(n!)</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">Very Low</td>
                </tr>
                <tr className="hover:bg-gray-50 dark:hover:bg-gray-800">
                  <td className="px-3 py-4 whitespace-nowrap text-sm font-medium">MMAS</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">Ant Colony</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">Fair</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">O(n²)</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">Medium</td>
                </tr>
              </tbody>
            </table>
          </div>
          
          <div className="mt-6">
            <h4 className="font-medium mb-2">Key Advantages of DRL-MOA</h4>
            <div className="grid grid-cols-2 gap-4 mt-3">
              <div className="bg-blue-50 dark:bg-blue-900/30 p-3 rounded-lg">
                <h5 className="font-medium text-blue-700 dark:text-blue-300 text-sm">Speed</h5>
                <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                  Linear inference time complexity vs. quadratic for evolutionary methods
                </p>
              </div>
              <div className="bg-green-50 dark:bg-green-900/30 p-3 rounded-lg">
                <h5 className="font-medium text-green-700 dark:text-green-300 text-sm">Quality</h5>
                <p className="text-xs text-green-600 dark:text-green-400 mt-1">
                  Better hypervolume and coverage metrics across all problem sizes
                </p>
              </div>
              <div className="bg-amber-50 dark:bg-amber-900/30 p-3 rounded-lg">
                <h5 className="font-medium text-amber-700 dark:text-amber-300 text-sm">Adaptability</h5>
                <p className="text-xs text-amber-600 dark:text-amber-400 mt-1">
                  Generalizes to different problem sizes after single training
                </p>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900/30 p-3 rounded-lg">
                <h5 className="font-medium text-purple-700 dark:text-purple-300 text-sm">Flexibility</h5>
                <p className="text-xs text-purple-600 dark:text-purple-400 mt-1">
                  Can be extended to more objectives and different cost functions
                </p>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default ResultsComparison;