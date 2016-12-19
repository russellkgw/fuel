require 'test_helper'

class ExchangeRateImportTest < ActiveSupport::TestCase
  test "valid case - insert and updatedata" do
    file = File.open(File.join(Rails.root, 'test','fixtures', 'files', 'exchange_rates', 'valid_exchange_rates.csv'))
    res = Data::ExchangeRateImport.import(file)
    assert_equal(true, res)
    file.close

    assert_equal(2, ExchangeRate.all.count)

    file = File.open(File.join(Rails.root, 'test','fixtures', 'files', 'exchange_rates', 'valid2_exchange_rates.csv'))
    res = Data::ExchangeRateImport.import(file)
    assert_equal(true, res)
    file.close

    assert_equal(8, ExchangeRate.find_by_date('2007-01-01').rate)
    assert_equal(6.945152, ExchangeRate.find_by_date('2007-01-02').rate)
  end

  test "invalid case - header invalid" do
    file = File.open(File.join(Rails.root, 'test','fixtures', 'files', 'exchange_rates', 'invalid_header_exchange_rates.csv'))
    res = Data::ExchangeRateImport.import(file)
    assert_equal(false, res)
    file.close

    file = File.open(File.join(Rails.root, 'test','fixtures', 'files', 'exchange_rates', 'invalid_header2_exchange_rates.csv'))
    res = Data::ExchangeRateImport.import(file)
    assert_equal(false, res)
    file.close

    file = File.open(File.join(Rails.root, 'test','fixtures', 'files', 'exchange_rates', 'invalid_header3_exchange_rates.csv'))
    res = Data::ExchangeRateImport.import(file)
    assert_equal(false, res)
    file.close
  end

  test "invalid case - invalid content" do
    file = File.open(File.join(Rails.root, 'test','fixtures', 'files', 'exchange_rates', 'invalid_content_exchange_rates.csv'))
    res = Data::ExchangeRateImport.import(file)
    assert_equal(false, res)
    file.close

    file = File.open(File.join(Rails.root, 'test','fixtures', 'files', 'exchange_rates', 'invalid_content2_exchange_rates.csv'))
    res = Data::ExchangeRateImport.import(file)
    assert_equal(false, res)
    file.close

    file = File.open(File.join(Rails.root, 'test','fixtures', 'files', 'exchange_rates', 'invalid_content3_exchange_rates.csv'))
    res = Data::ExchangeRateImport.import(file)
    assert_equal(false, res)
    file.close

    file = File.open(File.join(Rails.root, 'test','fixtures', 'files', 'exchange_rates', 'invalid_content4_exchange_rates.csv'))
    res = Data::ExchangeRateImport.import(file)
    assert_equal(false, res)
    file.close
  end

end