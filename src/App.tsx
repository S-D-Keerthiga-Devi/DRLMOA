import React, { useState } from 'react';
import { Brain, Network, Settings, BarChart, Map, Zap } from 'lucide-react';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import ModelTraining from './components/ModelTraining';
import ProblemGenerator from './components/ProblemGenerator';
import Visualization from './components/Visualization';
import DatasetLoader from './components/DatasetLoader';
import ResultsComparison from './components/ResultsComparison';
import './App.css';

export type TabType = 'dataset' | 'problem' | 'training' | 'visualization' | 'results' | 'settings';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('dataset');
  const [darkMode, setDarkMode] = useState(false);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  return (
    <div className={`min-h-screen flex flex-col ${darkMode ? 'dark bg-gray-900' : 'bg-gray-50'}`}>
      <Header darkMode={darkMode} toggleDarkMode={toggleDarkMode} />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          tabs={[
            { id: 'dataset', label: 'Dataset', icon: <Map size={20} /> },
            { id: 'problem', label: 'Problem Generator', icon: <Network size={20} /> },
            { id: 'training', label: 'Model Training', icon: <Brain size={20} /> },
            { id: 'visualization', label: 'Visualization', icon: <BarChart size={20} /> },
            { id: 'results', label: 'Results', icon: <Zap size={20} /> },
            { id: 'settings', label: 'Settings', icon: <Settings size={20} /> }
          ]}
        />
        
        <main className={`flex-1 p-6 overflow-auto ${darkMode ? 'text-white' : 'text-gray-800'}`}>
          {activeTab === 'dataset' && <DatasetLoader />}
          {activeTab === 'problem' && <ProblemGenerator />}
          {activeTab === 'training' && <ModelTraining />}
          {activeTab === 'visualization' && <Visualization />}
          {activeTab === 'results' && <ResultsComparison />}
          {activeTab === 'settings' && (
            <div className="p-6 max-w-4xl mx-auto">
              <h2 className="text-2xl font-bold mb-4">Settings</h2>
              <p className="mb-4">Configure the model parameters and training settings.</p>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;