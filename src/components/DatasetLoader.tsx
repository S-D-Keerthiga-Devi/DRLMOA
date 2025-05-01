import React, { useState } from 'react';
import { Upload, FileText, Check, X } from 'lucide-react';
import { Card } from './ui/Card';

interface DatasetFile {
  name: string;
  size: number;
  type: string;
  loaded: boolean;
}

const DatasetLoader: React.FC = () => {
  const [datasets, setDatasets] = useState<DatasetFile[]>([
    { name: 'kroA100.tsp', size: 21504, type: 'tsp', loaded: true },
    { name: 'kroB100.tsp', size: 21532, type: 'tsp', loaded: true },
    { name: 'kroA150.tsp', size: 32384, type: 'tsp', loaded: false },
    { name: 'kroB150.tsp', size: 32412, type: 'tsp', loaded: false },
    { name: 'kroA200.tsp', size: 43264, type: 'tsp', loaded: false },
    { name: 'kroB200.tsp', size: 43292, type: 'tsp', loaded: false },
  ]);

  const toggleDataset = (index: number) => {
    const newDatasets = [...datasets];
    newDatasets[index].loaded = !newDatasets[index].loaded;
    setDatasets(newDatasets);
  };

  const handleUpload = () => {
    // This would handle actual file upload in a real implementation
    alert('File upload would be implemented here in a real application');
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">Dataset Loader</h2>
      <p className="mb-6">
        Load standard TSPLIB datasets (kroA/kroB) for bi-objective TSP problems or upload your own datasets.
      </p>

      <div className="grid md:grid-cols-2 gap-6">
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Standard TSPLIB Datasets</h3>
          <div className="space-y-3">
            {datasets.map((dataset, index) => (
              <div 
                key={dataset.name} 
                className="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
              >
                <div className="flex items-center">
                  <FileText size={20} className="mr-3 text-blue-500" />
                  <div>
                    <p className="font-medium">{dataset.name}</p>
                    <p className="text-sm text-gray-500 dark:text-gray-400">{(dataset.size / 1024).toFixed(2)} KB</p>
                  </div>
                </div>
                <button 
                  className={`p-2 rounded-full ${dataset.loaded ? 'bg-green-100 text-green-600 dark:bg-green-900/30 dark:text-green-400' : 'bg-gray-100 text-gray-500 dark:bg-gray-700 dark:text-gray-400'}`}
                  onClick={() => toggleDataset(index)}
                >
                  {dataset.loaded ? <Check size={16} /> : <X size={16} />}
                </button>
              </div>
            ))}
          </div>
        </Card>

        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Upload Custom Dataset</h3>
          <div 
            className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-8 text-center hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors cursor-pointer"
            onClick={handleUpload}
          >
            <Upload size={36} className="mx-auto mb-4 text-blue-500" />
            <p className="font-medium mb-1">Click to upload or drag & drop</p>
            <p className="text-sm text-gray-500 dark:text-gray-400">Support for TSP, ATT, and GEO file formats</p>
          </div>
          
          <div className="mt-6">
            <h4 className="font-medium mb-2">Dataset Requirements:</h4>
            <ul className="list-disc pl-5 text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>For bi-objective Euclidean TSP: two separate location files</li>
              <li>For mixed bi-objective TSP: location file plus attribute file</li>
              <li>Maximum file size: 10MB</li>
            </ul>
          </div>
        </Card>
      </div>

      <div className="mt-8">
        <h3 className="text-lg font-semibold mb-4">Dataset Preview</h3>
        <div className="overflow-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">City ID</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">X (Obj 1)</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Y (Obj 1)</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">X (Obj 2)</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Y (Obj 2)</th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800">
              {[...Array(5)].map((_, i) => (
                <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">{i + 1}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">{(Math.random() * 1000).toFixed(2)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">{(Math.random() * 1000).toFixed(2)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">{(Math.random() * 1000).toFixed(2)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">{(Math.random() * 1000).toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default DatasetLoader;