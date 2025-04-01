import logging
logging.basicConfig(level=logging.INFO)

def recommend_options(price, prediction, risk='neutral', horizon='short'):
    """
    Recommend trading strategies based on predicted price changes.
    """
    try:
        if not isinstance(price, (int, float)) or not isinstance(prediction, (int, float)):
            raise ValueError('Price and prediction must be numeric values.')
        if risk not in ['neutral', 'hedge']:
            raise ValueError('Invalid risk preference. Choose from neutral or hedge.')
        if horizon not in ['short', 'long']:
            raise ValueError('Invalid horizon. Choose from short or long.')

        expected_change = prediction - price
        pct_change = expected_change / price if price != 0 else 0

        strategies = []

        if pct_change > 0.05:
            strategies.append('شراء Call')
            if horizon == 'short':
                strategies.append('شراء Vertical Call Spread')
        elif pct_change < -0.05:
            strategies.append('شراء Put')
            if horizon == 'short':
                strategies.append('شراء Vertical Put Spread')
        else:
            strategies.append('فتح Straddle أو Strangle')

        if risk == 'hedge':
            strategies.append('استخدام Covered Call أو Protective Put')

        return {
            'expected_change_pct': round(pct_change * 100, 2),
            'strategy': strategies
        }
    except Exception as e:
        logging.exception('Error during recommendation generation')
        return {
            'expected_change_pct': None,
            'strategy': ['Error: Unable to generate recommendation.']
        }
