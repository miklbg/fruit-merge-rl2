# Training Guide - Fruit Merge RL

This guide explains how to use the new training features that allow you to save training progress after each episode and continue training from a saved state.

## Features

### 1. Configurable Episode Count
You can now specify exactly how many episodes you want to train for (1-1000 episodes).

### 2. Automatic Save After Each Episode
The training system automatically saves your progress after every episode, so you never lose your work.

### 3. Continue Training from Saved State
You can stop training at any time and continue later from where you left off. The system remembers:
- The number of episodes completed
- The current learning rate
- All model weights

## How to Use

### Starting Training

1. Open the game and click the **"RL Controls"** button in the pause menu
2. Enter the number of episodes you want to train in the **"Number of Episodes"** field (default: 5)
3. Click **"Start Training"**
4. The training will run for the specified number of episodes, saving progress after each one

### Continuing Training

1. Click **"Load Model"** to restore your saved model and training state
2. The status will show your current episode count (e.g., "Model loaded (Episode 42)")
3. Enter how many **additional** episodes you want to train
4. Click **"Start Training"** to continue from where you left off

### Example Workflow

```
Day 1:
- Start new training with 10 episodes
- Training completes at Episode 10
- Model and metadata are saved

Day 2:
- Click "Load Model"
- Status shows "Model loaded (Episode 10)"
- Enter 20 more episodes
- Click "Start Training"
- Training continues from Episode 11 to Episode 30
```

## Training Progress

The system tracks the following metadata:
- **Episode Count**: Total number of episodes completed
- **Learning Rate**: Current learning rate (decays over time)
- **Timestamp**: When the training was last saved

This metadata is stored separately in localStorage under the key `fruit-merge-dqn-metadata`.

## Technical Details

### Save Locations
- Model weights: `localstorage://fruit-merge-dqn-v1`
- Training metadata: `localStorage['fruit-merge-dqn-metadata']`

### Learning Rate Decay
The learning rate automatically decays as training progresses:
- Episodes 0-99: 1e-4 (0.0001)
- Episodes 100-199: 5e-5 (0.00005)
- Episodes 200+: 3e-5 (0.00003)

When you continue training, the decay schedule picks up from where it left off based on your episode count.

### Performance Notes
- Saving after each episode may add a small delay (typically <100ms)
- This ensures you never lose progress even if the browser crashes
- Progress logs are printed every 5 episodes to avoid console spam

## Tips

1. **Start Small**: Begin with 5-10 episodes to test your setup
2. **Monitor Progress**: Watch the console logs to see training metrics
3. **Save Manually**: You can also click "Save Model" at any time to manually save
4. **Regular Backups**: Consider exporting your model periodically for safety

## Troubleshooting

**Q: My episode count reset to 0**
- Make sure you clicked "Load Model" before starting new training
- Check if localStorage was cleared (browser settings)

**Q: Training doesn't continue from saved state**
- Verify that "Model loaded successfully" appears in the status
- Check browser console for any error messages

**Q: Can I train on a different device?**
- Currently, models are stored in browser localStorage
- They are not synced across devices or browsers
- Future updates may add export/import functionality

## Related Files

- `game/train.js`: Training loop and model management
- `game/index.html`: UI and event handlers
- `localStorage`: Browser storage for model and metadata
