import React, { useState } from 'react';
import { Package, Search, Laptop as LaptopIcon, Filter } from 'lucide-react';
import { Product } from '../types';

interface CatalogueViewProps {
  products: Product[];
}

export const CatalogueView: React.FC<CatalogueViewProps> = ({ products }) => {
  const [search, setSearch] = useState('');
  const [maxBudget, setMaxBudget] = useState<number>(150000);

  const filteredProducts = products.filter((p) => {
    const matchesSearch =
      !search ||
      p.name.toLowerCase().includes(search.toLowerCase()) ||
      (p.processor && p.processor.toLowerCase().includes(search.toLowerCase())) ||
      (p.brand && p.brand.toLowerCase().includes(search.toLowerCase()));

    const matchesPrice = !maxBudget || p.price <= maxBudget;
    return matchesSearch && matchesPrice;
  });

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Package size={18} color="#2563EB" />
            <h1 style={styles.title}>Product Catalogue & Hardware Knowledge</h1>
          </div>
          <p style={styles.subtitle}>
            Active catalog searched and reasoned over by the AI agent during product recommendation
            traces.
          </p>
        </div>
      </div>

      {/* Filter Bar */}
      <div style={styles.filterBar}>
        <div style={styles.searchBox}>
          <Search size={14} color="var(--text-muted)" />
          <input
            type="text"
            placeholder="Search by laptop name, processor, brand..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={styles.searchInput}
          />
        </div>

        <div style={styles.budgetPills}>
          <span style={styles.budgetLabel}>Max Price:</span>
          {[50000, 75000, 100000, 150000].map((b) => (
            <button
              key={b}
              onClick={() => setMaxBudget(b)}
              style={{
                ...styles.budgetBtn,
                backgroundColor: maxBudget === b ? 'var(--bg-active)' : 'transparent',
                fontWeight: maxBudget === b ? 600 : 400,
                color: maxBudget === b ? 'var(--text-primary)' : 'var(--text-muted)',
              }}
            >
              ₹{(b / 1000).toFixed(0)}k
            </button>
          ))}
          <button
            onClick={() => setMaxBudget(999999)}
            style={{
              ...styles.budgetBtn,
              backgroundColor: maxBudget > 150000 ? 'var(--bg-active)' : 'transparent',
              fontWeight: maxBudget > 150000 ? 600 : 400,
              color: maxBudget > 150000 ? 'var(--text-primary)' : 'var(--text-muted)',
            }}
          >
            All
          </button>
        </div>
      </div>

      {/* Grid of Product Cards */}
      <div style={styles.productsGrid}>
        {filteredProducts.map((p, idx) => (
          <div key={idx} style={styles.card}>
            <div style={styles.imageBox}>
              {p.image_url ? (
                <img
                  src={p.image_url}
                  alt={p.name}
                  style={styles.image}
                  onError={(e) => {
                    e.currentTarget.style.display = 'none';
                    e.currentTarget.parentElement!.innerHTML =
                      '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#94A3B8;"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="2" y1="20" x2="22" y2="20"/></svg></div>';
                  }}
                />
              ) : (
                <div style={styles.placeholder}>
                  <LaptopIcon size={32} color="#94A3B8" />
                </div>
              )}
            </div>

            <div style={styles.info}>
              <div style={styles.name} title={p.name}>
                {p.name}
              </div>
              <div style={styles.price}>₹{p.price?.toLocaleString('en-IN')}</div>

              <div style={styles.specsRow}>
                <span>{p.ram ? `${p.ram} RAM` : '16GB RAM'}</span>
                <span>·</span>
                <span>{p.storage || '512GB SSD'}</span>
              </div>

              <div style={styles.cpuGpu}>
                {p.processor} {p.gpu ? `· ${p.gpu}` : ''}
              </div>

              <div style={styles.footerRow}>
                <span className="badge badge-success">✓ In Stock</span>
                {p.brand && <span style={styles.brandText}>{p.brand}</span>}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: '2rem',
    height: '100vh',
    overflowY: 'auto',
    backgroundColor: 'var(--bg-primary)',
    display: 'flex',
    flexDirection: 'column',
    gap: '1.25rem',
  },
  header: {
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
  },
  title: {
    fontSize: '1.4rem',
    fontWeight: 700,
    color: 'var(--text-primary)',
    letterSpacing: '-0.01em',
  },
  subtitle: {
    fontSize: '0.85rem',
    color: 'var(--text-secondary)',
    marginTop: '0.2rem',
  },
  filterBar: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '1rem',
  },
  searchBox: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.45rem',
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-color)',
    borderRadius: '6px',
    padding: '0.4rem 0.75rem',
    width: '320px',
  },
  searchInput: {
    border: 'none',
    backgroundColor: 'transparent',
    outline: 'none',
    width: '100%',
    fontSize: '0.84rem',
  },
  budgetPills: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.35rem',
  },
  budgetLabel: {
    fontSize: '0.75rem',
    color: 'var(--text-muted)',
    marginRight: '0.2rem',
  },
  budgetBtn: {
    padding: '0.3rem 0.6rem',
    borderRadius: '4px',
    border: '1px solid var(--border-color)',
    fontSize: '0.78rem',
  },
  productsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
    gap: '1rem',
  },
  card: {
    backgroundColor: 'var(--bg-primary)',
    border: '1px solid var(--border-color)',
    borderRadius: '8px',
    overflow: 'hidden',
    boxShadow: '0 1px 2px rgba(0,0,0,0.02)',
    display: 'flex',
    flexDirection: 'column',
  },
  imageBox: {
    height: '140px',
    backgroundColor: '#F8FAFC',
    borderBottom: '1px solid var(--border-color)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  image: {
    width: '100%',
    height: '100%',
    objectFit: 'cover',
  },
  placeholder: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
  },
  info: {
    padding: '0.85rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.25rem',
  },
  name: {
    fontSize: '0.88rem',
    fontWeight: 600,
    color: 'var(--text-primary)',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  price: {
    fontSize: '1rem',
    fontWeight: 700,
    color: 'var(--text-primary)',
  },
  specsRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.35rem',
    fontSize: '0.76rem',
    color: 'var(--text-secondary)',
  },
  cpuGpu: {
    fontSize: '0.73rem',
    color: 'var(--text-muted)',
  },
  footerRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: '0.5rem',
    paddingTop: '0.5rem',
    borderTop: '1px solid var(--border-color)',
  },
  brandText: {
    fontSize: '0.72rem',
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
  },
};
