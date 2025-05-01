import React from 'react';
import { Card } from './ui/Card';

export const ModelCode: React.FC = () => {
  const codeSnippets = [
    {
      title: 'DRL-MOA Implementation',
      code: `import torch
import torch.nn as nn
import torch.optim as optim

class DRLMOA(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DRLMOA, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        action = self.actor(state)
        value = self.critic(torch.cat([state, action], dim=1))
        return action, value

    def get_action(self, state):
        with torch.no_grad():
            action = self.actor(state)
        return action`
    },
    {
      title: 'Training Loop',
      code: `def train_drlmoa(env, model, episodes=1000):
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Get action from model
            action = model.get_action(state)
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Update model
            optimizer.zero_grad()
            current_action, current_value = model(state)
            _, next_value = model(next_state)
            
            # Compute TD error
            td_error = reward + (0.99 * next_value * (1 - done)) - current_value
            
            # Compute losses
            actor_loss = -current_value.mean()
            critic_loss = td_error.pow(2).mean()
            
            # Total loss
            loss = actor_loss + critic_loss
            
            # Backpropagate
            loss.backward()
            optimizer.step()
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
        print(f"Episode {episode}: Total Reward = {total_reward}")`
    }
  ];

  return (
    <div className="space-y-6 p-6">
      <h2 className="text-2xl font-bold mb-4">Model Implementation</h2>
      {codeSnippets.map((snippet, index) => (
        <Card key={index} className="overflow-hidden">
          <div className="bg-gray-100 px-4 py-2 border-b">
            <h3 className="text-lg font-semibold">{snippet.title}</h3>
          </div>
          <pre className="p-4 bg-gray-50 overflow-x-auto">
            <code className="text-sm">{snippet.code}</code>
          </pre>
        </Card>
      ))}
    </div>
  );
};