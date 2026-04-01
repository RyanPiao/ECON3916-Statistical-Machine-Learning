# 🍔 Global Purchasing Power Parity Analysis via the Big Mac Index

## Objective

This project applies the **Law of One Price** to real-world data by leveraging The Economist's Big Mac Index as a proxy for cross-border price arbitrage and currency valuation analysis.

## Methodology

- **Data Construction**: Manually constructed a structured dataset from The Economist's 2015 Big Mac Index using Python dictionaries, capturing local Big Mac prices and nominal exchange rates across multiple countries.

- **Implied PPP Calculation**: Computed the implied purchasing power parity (PPP) exchange rate for each currency by comparing the local price of a Big Mac to the U.S. benchmark price.

- **Currency Valuation Assessment**: Derived percentage over/undervaluation metrics by comparing the implied PPP exchange rate to the actual nominal exchange rate, identifying potential arbitrage opportunities.

## Key Findings

The analysis revealed significant currency misalignments relative to the U.S. dollar. For example, a Big Mac priced at **$5.69 in New York** versus **$3.50 in Shanghai** suggests the **Chinese yuan is undervalued by approximately 38%** against the dollar, assuming transportation costs and market frictions are negligible.

This deviation from purchasing power parity indicates potential arbitrage opportunities and reflects underlying economic factors such as labor cost differentials, trade barriers, and productivity gaps. Currencies trading below their implied PPP values may signal competitive export advantages, while overvalued currencies could indicate inflationary pressures or stronger domestic purchasing power.

## Economic Significance

The Big Mac Index, while informal, serves as an accessible benchmark for understanding exchange rate dynamics and testing fundamental economic theories in international finance. This "burgernomics" approach provides intuitive insights into complex phenomena like currency misalignment and long-run equilibrium exchange rates.

---

**Tools Used**: Python, Pandas  
**Data Source**: The Economist (2015)
