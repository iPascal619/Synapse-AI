# Investigation Dashboard - Synapse AI Fraud Detection

## Overview
React/Next.js web application for fraud analysts to review flagged transactions, provide feedback, and monitor system performance.

## Features

### Transaction Review
- **Review Queue**: List of transactions flagged for manual review
- **Transaction Details**: Complete view of transaction data and computed features
- **SHAP Explanations**: Visual feature importance for model decisions
- **Historical Context**: User's transaction history and behavior patterns

### Analyst Workflow
- **Decision Making**: Approve/Deny/Escalate flagged transactions
- **Feedback Collection**: Label transactions as fraud/legitimate for model training
- **Case Notes**: Add investigation notes and reasoning
- **Batch Operations**: Process multiple similar cases efficiently

### Analytics & Monitoring
- **Performance Dashboard**: Model accuracy, false positive rates, processing times
- **Fraud Trends**: Pattern analysis and emerging threat detection
- **Alert Management**: System alerts and investigation priorities
- **Reporting**: Export capabilities for compliance and auditing

## Technology Stack

- **Framework**: Next.js 13+ with App Router
- **UI Library**: Tailwind CSS + shadcn/ui components
- **Charts**: Recharts for data visualization
- **State Management**: Zustand
- **API Client**: Axios with React Query
- **Authentication**: NextAuth.js
- **Database**: PostgreSQL for user management and case data

## Key Components

### 1. Review Dashboard
```typescript
// Main dashboard showing pending reviews
<ReviewDashboard />
  <TransactionTable />
  <FilterControls />
  <PaginationControls />
```

### 2. Transaction Detail View
```typescript
// Detailed transaction analysis
<TransactionDetail id={transactionId} />
  <TransactionInfo />
  <FeatureExplanation />
  <SHAPVisualization />
  <DecisionPanel />
```

### 3. Analytics Dashboard
```typescript
// System performance monitoring
<AnalyticsDashboard />
  <MetricsGrid />
  <FraudTrends />
  <ModelPerformance />
  <AlertsPanel />
```

## API Integration

### Endpoints Used
- `GET /api/transactions/review` - Get transactions for review
- `POST /api/transactions/{id}/decision` - Submit analyst decision
- `GET /api/analytics/performance` - Get system metrics
- `GET /api/shap/explanation/{id}` - Get SHAP explanation

### Real-time Updates
- WebSocket connection for live transaction updates
- Server-sent events for system alerts
- Optimistic UI updates for better UX

## Setup Instructions

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env.local
   # Edit .env.local with your configuration
   ```

3. **Run Development Server**
   ```bash
   npm run dev
   ```

4. **Build for Production**
   ```bash
   npm run build
   npm start
   ```

## Security

- **Authentication**: Role-based access control
- **Authorization**: Analyst permissions and data access controls
- **Audit Trail**: All decisions and actions logged
- **Data Protection**: PII masking and encryption

## Performance

- **Client-side Caching**: React Query for API response caching
- **Server-side Rendering**: Next.js SSR for fast initial loads
- **Code Splitting**: Automatic route-based code splitting
- **Image Optimization**: Next.js Image component
